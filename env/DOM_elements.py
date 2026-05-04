"""
DOM_elements.py

Extracts every DOM element from a live webpage.

Each element is stored as a DOMNode dataclass containing:
  - tag name
  - text content
  - bounding box (x, y, width, height) in CSS pixels
  - all HTML attributes
  - XPath  (for precise re-location)
  - whether the element is interactable (clickable / typeable)

Dependencies:
    pip install selenium
    pip install chromedriver-autoinstaller
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webelement import WebElement
import chromedriver_autoinstaller


# ── Data structure for one DOM node ───────────────────────────────────────────

@dataclass
class DOMNode:
    """
    Represents a single element extracted from the page DOM.

    Attributes
    ----------
    tag : str
        Lower-case HTML tag name (e.g. 'a', 'button', 'input').
    text : str
        Visible text content (stripped).
    xpath : str
        Absolute XPath that uniquely identifies the element.
    attributes : dict[str, str]
        All HTML attributes as key-value pairs.
    bbox : dict  { x, y, width, height }
        Bounding box in CSS pixels relative to the viewport.
        x / y are the top-left corner.
    is_clickable : bool
        True for <a>, <button>, <select>, role="button", etc.
    is_typeable : bool
        True for <input>, <textarea>, contenteditable elements.
    is_visible : bool
        False if the element has zero size or is hidden.
    """
    tag:          str
    text:         str
    xpath:        str
    attributes:   dict[str, str]           = field(default_factory=dict)
    bbox:         dict[str, float]         = field(default_factory=dict)
    is_clickable: bool                     = False
    is_typeable:  bool                     = False
    is_visible:   bool                     = True

    def __repr__(self) -> str:
        return (
            f"DOMNode(tag={self.tag!r}, text={self.text[:40]!r}, "
            f"bbox={self.bbox}, clickable={self.is_clickable}, "
            f"typeable={self.is_typeable})"
        )


# ── Main class ────────────────────────────────────────────────────────────────

class dom_elements:
    """
    Opens a URL in a headless Chrome browser and extracts every DOM element.

    Parameters
    ----------
    url : str
        The fully-qualified URL to inspect.
    wait_seconds : float
        Time to wait after page load for JS to settle. Default 2.
    viewport_width : int
        Browser viewport width in pixels. Default 1280.
    viewport_height : int
        Browser viewport height in pixels. Default 900.

    Usage
    -----
    >>> extractor = dom_elements("https://example.com")
    >>> nodes = extractor.extract()
    >>> print(len(nodes), "elements found")
    >>> clickable = extractor.filter(is_clickable=True)
    """

    # Tags whose elements are interactive by nature
    _CLICKABLE_TAGS  = {"a", "button", "select", "option", "label", "summary"}
    _TYPEABLE_TAGS   = {"input", "textarea"}
    _SKIP_TAGS       = {
        "script", "style", "noscript", "meta", "head",
        "link", "br", "hr", "path", "svg", "defs",
    }

    def __init__(
        self,
        url: str,
        wait_seconds: float = 2.0,
        viewport_width: int = 1280,
        viewport_height: int = 900,
    ):
        self.url              = url
        self.wait_seconds     = wait_seconds
        self.viewport_width   = viewport_width
        self.viewport_height  = viewport_height
        self._nodes: list[DOMNode] = []
        self._driver: Optional[webdriver.Chrome] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self) -> list[DOMNode]:
        """
        Navigate to the URL and extract all DOM elements.

        Returns
        -------
        list[DOMNode]
            Every element found in the page DOM.
        """
        self._nodes = []
        self._launch_driver()
        try:
            self._driver.get(self.url)
            time.sleep(self.wait_seconds)
            all_elements: list[WebElement] = self._driver.find_elements(
                "xpath", "//*"
            )
            for elem in all_elements:
                node = self._parse_element(elem)
                if node is not None:
                    self._nodes.append(node)
        finally:
            self._driver.quit()
            self._driver = None

        print(f"[dom_elements] Extracted {len(self._nodes)} nodes from {self.url}")
        return self._nodes

    def filter(
        self,
        tag: Optional[str] = None,
        is_clickable: Optional[bool] = None,
        is_typeable: Optional[bool] = None,
        is_visible: Optional[bool] = True,
        text_contains: Optional[str] = None,
    ) -> list[DOMNode]:
        """
        Return a filtered subset of the extracted nodes.

        All provided filters are combined with AND logic.
        """
        result = self._nodes
        if tag            is not None: result = [n for n in result if n.tag == tag.lower()]
        if is_clickable   is not None: result = [n for n in result if n.is_clickable == is_clickable]
        if is_typeable    is not None: result = [n for n in result if n.is_typeable  == is_typeable]
        if is_visible     is not None: result = [n for n in result if n.is_visible   == is_visible]
        if text_contains  is not None:
            result = [n for n in result if text_contains.lower() in n.text.lower()]
        return result

    @property
    def nodes(self) -> list[DOMNode]:
        """All extracted DOMNode objects."""
        return self._nodes

    def summary(self) -> dict[str, int]:
        """Return a tag-frequency dictionary."""
        freq: dict[str, int] = {}
        for n in self._nodes:
            freq[n.tag] = freq.get(n.tag, 0) + 1
        return dict(sorted(freq.items(), key=lambda kv: -kv[1]))

    # ── Private helpers ────────────────────────────────────────────────────────

    def _launch_driver(self):
        chromedriver_autoinstaller.install()
        opts = Options()
        opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument(f"--window-size={self.viewport_width},{self.viewport_height}")
        self._driver = webdriver.Chrome(options=opts)

    def _parse_element(self, elem: WebElement) -> Optional[DOMNode]:
        """Convert a Selenium WebElement into a DOMNode, or return None to skip."""
        try:
            tag = elem.tag_name.lower()
        except Exception:
            return None

        if tag in self._SKIP_TAGS:
            return None

        # Bounding box
        try:
            rect = elem.rect          # {'x', 'y', 'width', 'height'}
        except Exception:
            rect = {"x": 0, "y": 0, "width": 0, "height": 0}

        is_visible = rect["width"] > 0 and rect["height"] > 0

        # Text content
        try:
            text = (elem.text or "").strip()
        except Exception:
            text = ""

        # HTML attributes
        try:
            attrs = self._driver.execute_script(
                """
                var items = {};
                for (var i = 0; i < arguments[0].attributes.length; i++) {
                    items[arguments[0].attributes[i].name] =
                        arguments[0].attributes[i].value;
                }
                return items;
                """,
                elem,
            ) or {}
        except Exception:
            attrs = {}

        # XPath
        try:
            xpath = self._driver.execute_script(
                """
                function getXPath(el) {
                    if (el.id) return '//*[@id="' + el.id + '"]';
                    var parts = [];
                    while (el && el.nodeType === 1) {
                        var idx = 1;
                        var sib = el.previousSibling;
                        while (sib) {
                            if (sib.nodeType === 1 && sib.tagName === el.tagName) idx++;
                            sib = sib.previousSibling;
                        }
                        parts.unshift(el.tagName.toLowerCase() + '[' + idx + ']');
                        el = el.parentNode;
                    }
                    return '/' + parts.join('/');
                }
                return getXPath(arguments[0]);
                """,
                elem,
            ) or ""
        except Exception:
            xpath = ""

        # Interactability
        role       = attrs.get("role", "").lower()
        input_type = attrs.get("type", "").lower()
        is_clickable = (
            tag in self._CLICKABLE_TAGS
            or role in {"button", "link", "checkbox", "radio", "menuitem", "tab"}
            or attrs.get("onclick") is not None
        )
        is_typeable = (
            tag in self._TYPEABLE_TAGS
            or attrs.get("contenteditable", "false").lower() in {"true", ""}
        ) and input_type not in {"submit", "button", "reset", "image"}

        return DOMNode(
            tag          = tag,
            text         = text,
            xpath        = xpath,
            attributes   = attrs,
            bbox         = rect,
            is_clickable = is_clickable,
            is_typeable  = is_typeable,
            is_visible   = is_visible,
        )
