"""
dom_feature_extractor.py

Matches a natural-language query against interactive DOM elements extracted
live from a Playwright ``Page`` object, then builds a 2-D text-feature map
over the viewport where higher activations mark elements that best match
the query.

* ``DOMElement`` are scraped from the live page via ``extract_dom(page)``.
* Bounding boxes come from Playwright's ``element.bounding_box()`` API.
* Only *visible*, *interactive* elements are considered (links, buttons,
  inputs, selects, textareas, labelled elements).
* The feature map dimensions default to the viewport size and are updated
  automatically when ``extract()`` is called with a ``page`` argument.
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class DOMElement:
    """
    A single interactive / text-bearing element from the live DOM.

    Attributes
    ----------
    text : str
        Visible label, placeholder, aria-label, or inner text.
    x, y : float
        Top-left corner of the bounding box in viewport pixels.
    w, h : float
        Bounding-box width and height in viewport pixels.
    tag  : str
        HTML tag name (lower-case), e.g. ``"button"``, ``"input"``.
    selector : str
        A CSS selector or XPath string that can be used to re-locate
        the element on the page for action execution.
    """
    text:     str
    x:        float
    y:        float
    w:        float
    h:        float
    tag:      str   = "unknown"
    selector: str   = ""

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


# CSS selector that targets every element a user might interact with
_INTERACTIVE_SELECTOR = (
    "a[href], button, input, select, textarea, "
    "[role='button'], [role='link'], [role='checkbox'], "
    "[role='radio'], [role='menuitem'], [role='tab'], "
    "[tabindex]:not([tabindex='-1'])"
)


class DOMFeatureExtractor:
    """
    Builds a query-specific 2-D feature map over the browser viewport by
    scraping interactive DOM elements from a Playwright ``Page`` and scoring
    each one against the query using normalised edit-distance similarity.

    Parameters
    ----------
    viewport_width  : int
        Width of the feature map (should equal the Playwright viewport width).
    viewport_height : int
        Height of the feature map.
    score_threshold : float
        Minimum similarity score ``[0, 1]`` to place in the map.
    max_elements    : int
        Cap on the number of DOM elements to score (for performance).
    """

    def __init__(
        self,
        viewport_width:  int   = 1280,
        viewport_height: int   = 720,
        score_threshold: float = 0.0,
        max_elements:    int   = 200,
    ) -> None:
        self.viewport_width  = viewport_width
        self.viewport_height = viewport_height
        self.score_threshold = score_threshold
        self.max_elements    = max_elements

    # ------------------------------------------------------------------
    # DOM scraping (Playwright integration)
    # ------------------------------------------------------------------

    def extract_dom(self, page) -> List[DOMElement]:
        """
        Scrape all visible interactive elements from a live Playwright Page.

        Parameters
        ----------
        page : playwright.sync_api.Page  (or async equivalent)
            An open Playwright page at the target URL.

        Returns
        -------
        list of DOMElement
            Every visible, interactive element with its bounding box and
            best available text label.
        """
        elements: List[DOMElement] = []
        handles = page.query_selector_all(_INTERACTIVE_SELECTOR)

        for i, handle in enumerate(handles):
            if i >= self.max_elements:
                break
            try:
                if not handle.is_visible():
                    continue
                bbox = handle.bounding_box()
                if bbox is None or bbox["width"] == 0 or bbox["height"] == 0:
                    continue

                tag   = (handle.evaluate("el => el.tagName") or "").lower()
                label = self._best_label(handle)
                if not label:
                    continue

                elements.append(DOMElement(
                    text     = label,
                    x        = bbox["x"],
                    y        = bbox["y"],
                    w        = bbox["width"],
                    h        = bbox["height"],
                    tag      = tag,
                    selector = self._unique_selector(handle, i),
                ))
            except Exception:
                # Element may have detached between query and access
                continue

        return elements

    # ------------------------------------------------------------------
    # Feature map construction
    # ------------------------------------------------------------------

    def extract(
        self,
        query:       str,
        dom:         Sequence[DOMElement],
        map_width:   int | None = None,
        map_height:  int | None = None,
    ) -> np.ndarray:
        """
        Build a 2-D query-specific feature map from a list of DOM elements.

        Parameters
        ----------
        query      : natural-language task instruction.
        dom        : DOM elements (from ``extract_dom`` or supplied manually).
        map_width  : override feature map width (defaults to viewport_width).
        map_height : override feature map height (defaults to viewport_height).

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(map_height, map_width)`` where higher
            values mark elements that match the query.
        """
        w = map_width  or self.viewport_width
        h = map_height or self.viewport_height
        feature_map = np.zeros((h, w), dtype=np.float32)

        query_words = self._tokenize(query)
        if not query_words:
            return feature_map

        for element in dom:
            score = self._element_score(query_words, element.text)
            if score <= self.score_threshold:
                continue
            cx, cy = element.center
            col = int(np.clip(round(cx), 0, w - 1))
            row = int(np.clip(round(cy), 0, h - 1))
            if score > feature_map[row, col]:
                feature_map[row, col] = score

        return feature_map

    def extract_from_page(self, query: str, page) -> Tuple[np.ndarray, List[DOMElement]]:
        """
        Convenience method: scrape DOM from a live Playwright page *and*
        build the feature map in one call.

        Returns
        -------
        feature_map : np.ndarray  shape (H, W)
        dom_elements : list[DOMElement]  (re-used downstream for action targeting)
        """
        dom_elements = self.extract_dom(page)
        feature_map  = self.extract(query, dom_elements)
        return feature_map, dom_elements

    # ------------------------------------------------------------------
    # Label extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best_label(handle) -> str:
        """
        Return the richest available text label for an element, trying
        (in order): aria-label → placeholder → value → inner text.
        """
        for attr in ("aria-label", "placeholder", "title", "alt", "value"):
            val = handle.get_attribute(attr)
            if val and val.strip():
                return val.strip()
        text = (handle.inner_text() or "").strip()
        return text[:120]   # cap length to avoid noisy long strings

    @staticmethod
    def _unique_selector(handle, index: int) -> str:
        """
        Derive a CSS selector that can re-locate this element.
        Falls back to a data-index attribute used by EnvironmentHandler.
        """
        try:
            sel = handle.evaluate()
            if sel:
                return sel
        except Exception:
            pass
        return f"[data-pw-index='{index}']"

    # ------------------------------------------------------------------
    # Similarity scoring
    # ------------------------------------------------------------------

    def _element_score(self, query_words: List[str], element_text: str) -> float:
        element_words = self._tokenize(element_text)
        if not element_words:
            return 0.0
        best = 0.0
        for qw in query_words:
            for ew in element_words:
                sim = self._edit_similarity(qw, ew)
                if sim > best:
                    best = sim
        return best

    @staticmethod
    def _edit_similarity(a: str, b: str) -> float:
        a, b = a.lower(), b.lower()
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0
        m, n   = len(a), len(b)
        prev   = list(range(n + 1))
        for i, ca in enumerate(a, 1):
            curr = [i] + [0] * n
            for j, cb in enumerate(b, 1):
                curr[j] = min(
                    prev[j]     + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (ca != cb),
                )
            prev = curr
        return 1.0 - prev[n] / max(m, n)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok for tok in re.split(r"\W+", text.lower()) if tok]

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DOMFeatureExtractor("
            f"viewport={self.viewport_width}×{self.viewport_height}, "
            f"threshold={self.score_threshold}, "
            f"max_elements={self.max_elements})"
        )
