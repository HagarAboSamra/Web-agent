"""
MiniWOB++ Observation Space
===========================
Flexible observation space for any MiniWOB task.
Returns:
  - screenshot: (210, 160, 3) uint8 RGB image
  - dom_elements: list of structured DOM element dicts
  - query: task instruction string
  - text_feature_map: query–DOM matching feature matrix
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────
# DOM element dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class DOMElement:
    """
    A single DOM node extracted from the MiniWOB environment.
    All spatial coords are in pixel space of the 210×160 viewport.
    """
    ref: int                          # unique node id
    tag: str                          # e.g. "button", "input", "div"
    text: str                         # visible text content
    value: str                        # input value if applicable
    placeholder: str                  # placeholder attribute
    classes: list[str]                # CSS class list
    bbox: tuple[float, float, float, float]  # (left, top, right, bottom)
    is_interactable: bool             # clickable / typeable?
    depth: int                        # DOM tree depth
    attrs: dict[str, str] = field(default_factory=dict)   # raw HTML attrs

    # ── derived helpers ───────────────────────────────────────
    @property
    def center(self) -> tuple[float, float]:
        l, t, r, b = self.bbox
        return ((l + r) / 2, (t + b) / 2)

    @property
    def area(self) -> float:
        l, t, r, b = self.bbox
        return max(0.0, r - l) * max(0.0, b - t)

    def to_text(self) -> str:
        """Concatenate all textual signals for embedding / matching."""
        parts = [self.tag, self.text, self.value, self.placeholder]
        parts += self.classes
        parts += [f"{k}={v}" for k, v in self.attrs.items()]
        return " ".join(p for p in parts if p).lower()


# ─────────────────────────────────────────────────────────────
# Text feature map: query × DOM matching
# ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    """Term-frequency vector (normalised)."""
    counts: dict[str, float] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0.0) + 1.0
    total = max(len(tokens), 1)
    return {k: v / total for k, v in counts.items()}


def compute_text_feature_map(
    query: str,
    dom_elements: list[DOMElement],
    max_elements: int = 64,
) -> np.ndarray:
    """
    Compute a (max_elements, F) text feature matrix where each row
    describes how well a DOM element matches the query.

    Feature columns per element (F = 8):
      0  exact_word_overlap        – fraction of query tokens found in element
      1  element_coverage          – fraction of element tokens found in query
      2  char_jaccard              – Jaccard on character n-grams (n=3)
      3  prefix_match              – any query token is prefix of element token
      4  tag_score                 – interactable tag bonus
      5  norm_area                 – bounding-box area / total viewport area
      6  norm_cx                   – normalised x centre [0,1]
      7  norm_cy                   – normalised y centre [0,1]

    Rows beyond len(dom_elements) are zero-padded.
    """
    VIEWPORT_AREA = 210.0 * 160.0
    F = 8

    out = np.zeros((max_elements, F), dtype=np.float32)

    q_tokens = set(_tokenize(query))
    q_ngrams = _char_ngrams(query, n=3)

    INTERACTABLE_TAGS = {"button", "input", "select", "textarea", "a", "option"}

    for i, elem in enumerate(dom_elements[:max_elements]):
        elem_text = elem.to_text()
        e_tokens = set(_tokenize(elem_text))
        e_ngrams = _char_ngrams(elem_text, n=3)

        # 0 – exact word overlap (query → element)
        overlap = len(q_tokens & e_tokens)
        out[i, 0] = overlap / max(len(q_tokens), 1)

        # 1 – element coverage (element → query)
        out[i, 1] = overlap / max(len(e_tokens), 1)

        # 2 – character Jaccard
        union = len(q_ngrams | e_ngrams)
        out[i, 2] = len(q_ngrams & e_ngrams) / max(union, 1)

        # 3 – prefix match
        out[i, 3] = float(
            any(et.startswith(qt) or qt.startswith(et)
                for qt in q_tokens for et in e_tokens)
        )

        # 4 – interactable tag score
        out[i, 4] = float(elem.tag.lower() in INTERACTABLE_TAGS)

        # 5-7 – spatial
        cx, cy = elem.center
        out[i, 5] = float(np.clip(elem.area / VIEWPORT_AREA, 0, 1))
        out[i, 6] = float(np.clip(cx / 210.0, 0, 1))
        out[i, 7] = float(np.clip(cy / 160.0, 0, 1))

    return out   # shape: (max_elements, 8)


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    text = re.sub(r"\s+", " ", text.lower())
    return {text[i : i + n] for i in range(len(text) - n + 1)}


# ─────────────────────────────────────────────────────────────
# Observation space
# ─────────────────────────────────────────────────────────────

class MiniWOBObservationSpace:
    """
    Flexible observation space for *any* MiniWOB++ task.

    Parameters
    ----------
    task_name : str
        MiniWOB task name, e.g. "click-button", "login-user".
    task_args : dict, optional
        Extra keyword arguments forwarded to the task (seeds, configs, …).
    max_dom_elements : int
        Maximum number of DOM elements kept per step.
    img_height, img_width : int
        Screenshot dimensions (default matches MiniWOB viewport 160×210).

    Observation dict keys
    ---------------------
    "screenshot"        np.ndarray  (H, W, 3)  uint8
    "dom_elements"      list[DOMElement]
    "query"             str                     natural-language task instruction
    "text_feature_map"  np.ndarray  (max_dom_elements, 8)  float32
    """

    SCREENSHOT_SHAPE = (160, 210, 3)   # H × W × C

    def __init__(
        self,
        task_name: str,
        task_args: dict[str, Any] | None = None,
        max_dom_elements: int = 64,
        img_height: int = 160,
        img_width: int = 210,
    ) -> None:
        self.task_name = task_name
        self.task_args = task_args or {}
        self.max_dom_elements = max_dom_elements
        self.img_height = img_height
        self.img_width = img_width

    # ── public interface ──────────────────────────────────────

    def observe(self, raw_env_state: dict[str, Any]) -> dict[str, Any]:
        """
        Transform a raw MiniWOB environment state into a structured
        observation dict.

        Parameters
        ----------
        raw_env_state : dict
            Must contain:
              "screenshot"  – PIL Image or np.ndarray (H,W,3) uint8
              "dom_info"    – MiniWOB DOMInfo / list of element dicts
              "utterance"   – task instruction string

        Returns
        -------
        dict with keys: screenshot, dom_elements, query, text_feature_map
        """
        screenshot   = self._process_screenshot(raw_env_state["screenshot"])
        dom_elements = self._parse_dom(raw_env_state["dom_info"])
        query        = raw_env_state.get("utterance", "")

        text_feature_map = compute_text_feature_map(
            query, dom_elements, self.max_dom_elements
        )

        return {
            "screenshot":       screenshot,         # (H, W, 3) uint8
            "dom_elements":     dom_elements,        # list[DOMElement]
            "query":            query,               # str
            "text_feature_map": text_feature_map,   # (max_dom, 8) float32
        }

    @property
    def observation_spec(self) -> dict[str, Any]:
        """Human-readable description of the observation space."""
        return {
            "screenshot":       f"np.uint8  ({self.img_height}, {self.img_width}, 3)",
            "dom_elements":     f"list[DOMElement]  max_len={self.max_dom_elements}",
            "query":            "str",
            "text_feature_map": f"np.float32  ({self.max_dom_elements}, 8)",
        }

    # ── private helpers ───────────────────────────────────────

    def _process_screenshot(self, raw: Any) -> np.ndarray:
        """Accept PIL Image or ndarray; resize to (H, W, 3) uint8."""
        try:
            import PIL.Image as PILImage

            if isinstance(raw, PILImage.Image):
                img = raw.convert("RGB").resize(
                    (self.img_width, self.img_height), PILImage.BILINEAR
                )
                return np.array(img, dtype=np.uint8)
        except ImportError:
            pass

        if isinstance(raw, np.ndarray):
            arr = raw
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape != (self.img_height, self.img_width, 3):
                # best-effort resize via slicing / zero-pad
                arr = _simple_resize(arr, self.img_height, self.img_width)
            return arr

        # Fallback: blank frame
        return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

    def _parse_dom(self, dom_info: Any) -> list[DOMElement]:
        """
        Convert MiniWOB DOMInfo (or list of dicts) to DOMElement objects.
        Works with both the official gym-miniwob DOMInfo namedtuple and
        raw dict representations from selenium-based wrappers.
        """
        if dom_info is None:
            return []

        elements: list[DOMElement] = []

        # Handle list-of-dicts (e.g. from selenium scraper)
        if isinstance(dom_info, list):
            for item in dom_info:
                elem = self._dict_to_dom_element(item)
                if elem is not None:
                    elements.append(elem)
            return elements

        # Handle object with .elements attribute (gym-miniwob DOMInfo)
        raw_list = getattr(dom_info, "elements", None)
        if raw_list is not None:
            for item in raw_list:
                elem = self._obj_to_dom_element(item)
                if elem is not None:
                    elements.append(elem)

        return elements

    @staticmethod
    def _dict_to_dom_element(d: dict) -> DOMElement | None:
        try:
            bbox_raw = d.get("bbox", d.get("rect", [0, 0, 0, 0]))
            if isinstance(bbox_raw, dict):
                bbox = (
                    bbox_raw.get("left", 0),
                    bbox_raw.get("top", 0),
                    bbox_raw.get("right", 0),
                    bbox_raw.get("bottom", 0),
                )
            else:
                bbox = tuple(bbox_raw[:4]) if len(bbox_raw) >= 4 else (0, 0, 0, 0)

            INTERACTABLE = {"button", "input", "select", "textarea", "a", "option"}
            tag = str(d.get("tag", d.get("tagName", "div"))).lower()

            return DOMElement(
                ref=int(d.get("ref", d.get("id", 0))),
                tag=tag,
                text=str(d.get("text", d.get("textContent", ""))).strip(),
                value=str(d.get("value", "")),
                placeholder=str(d.get("placeholder", "")),
                classes=_split_classes(d.get("classes", d.get("className", ""))),
                bbox=bbox,
                is_interactable=tag in INTERACTABLE or bool(d.get("interactable", False)),
                depth=int(d.get("depth", 0)),
                attrs={k: str(v) for k, v in d.get("attrs", {}).items()},
            )
        except Exception:
            return None

    @staticmethod
    def _obj_to_dom_element(obj: Any) -> DOMElement | None:
        """Parse a gym-miniwob DOMElement namedtuple / object."""
        try:
            rect = getattr(obj, "rect", None) or getattr(obj, "bbox", None)
            if rect is not None and hasattr(rect, "left"):
                bbox = (rect.left, rect.top, rect.right, rect.bottom)
            elif rect is not None:
                bbox = tuple(rect)[:4]
            else:
                bbox = (0.0, 0.0, 0.0, 0.0)

            tag = str(getattr(obj, "tag", "div")).lower()
            INTERACTABLE = {"button", "input", "select", "textarea", "a", "option"}

            return DOMElement(
                ref=int(getattr(obj, "ref", 0)),
                tag=tag,
                text=str(getattr(obj, "text", "")).strip(),
                value=str(getattr(obj, "value", "")),
                placeholder=str(getattr(obj, "placeholder", "")),
                classes=_split_classes(getattr(obj, "classes", "")),
                bbox=bbox,
                is_interactable=tag in INTERACTABLE,
                depth=int(getattr(obj, "depth", 0)),
                attrs=dict(getattr(obj, "attrs", {})),
            )
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────

def _split_classes(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(c) for c in raw]
    return str(raw).split() if raw else []


def _simple_resize(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize without PIL dependency."""
    src_h, src_w = arr.shape[:2]
    row_idx = (np.arange(h) * src_h / h).astype(int)
    col_idx = (np.arange(w) * src_w / w).astype(int)
    return arr[np.ix_(row_idx, col_idx)]


if __name__ == "__main__":
    import time
    import gymnasium
    import miniwob
    import miniwob.envs

    gymnasium.register_envs(miniwob)

    print("=" * 60)
    print("ObservationSpace — REAL MiniWoB test")
    print("=" * 60)

    # --------------------------------------------------
    # Instantiate the observation space
    # --------------------------------------------------
    obs_space = MiniWOBObservationSpace(
        task_name="click-test-2",
        max_dom_elements=64,
    )

    print(obs_space.observation_spec)
    print()

    # --------------------------------------------------
    # Create environment
    # --------------------------------------------------
    env = gymnasium.make("miniwob/click-test-2-v1", render_mode=None)

    try:
        raw_obs, _ = env.reset()

        # --------------------------------------------------
        # raw_obs already contains "screenshot", "dom_elements"
        # (gym-miniwob objects), and "utterance".
        # obs_space.observe() handles the conversion internally.
        # --------------------------------------------------
        t0 = time.perf_counter()

        out = obs_space.observe({
            "screenshot": raw_obs["screenshot"],
            "dom_info":   raw_obs["dom_elements"],   # list of gym-miniwob DOM objects
            "utterance":  raw_obs.get("utterance", ""),
        })

        elapsed = time.perf_counter() - t0

        screenshot       = out["screenshot"]         # np.ndarray (H, W, 3) uint8
        dom_elements     = out["dom_elements"]       # list[DOMElement]
        query            = out["query"]              # str
        text_feature_map = out["text_feature_map"]   # (max_dom, 8) float32

        print(f"screenshot shape   : {screenshot.shape}")
        print(f"dom_elements count : {len(dom_elements)}")
        print(f"query              : {query!r}")
        print(f"text_feature_map   : {text_feature_map.shape}")
        print(f"\nInference time     : {elapsed * 1000:.1f} ms")

        # --------------------------------------------------
        # Per-element scores (column 0: query–word overlap)
        # --------------------------------------------------
        print("\nPer-element query-overlap scores:")

        for i, el in enumerate(dom_elements):
            score = text_feature_map[i, 0]           # exact word overlap feature
            text  = el.text if el.text else ""
            print(
                f"  [{el.tag:8s}] "
                f"'{text[:20]:20s}' "
                f"overlap={score:.4f}"
            )

        print("\n✓ Real MiniWoB smoke test passed.")

    finally:
        env.close()