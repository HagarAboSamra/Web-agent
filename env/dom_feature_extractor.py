"""
dom_feature_extractor.py
Builds a query-relevance heatmap from the DOM fields provided by the
MiniWoB++ Gymnasium API (info["dom_elements"] list from env.step / env.reset).

Each DOM element dict has keys:
    ref, tag, text, value, id, classes, focused, tampered,
    left, top, width, height   (all in pixels relative to viewport)

No Playwright required.
"""

import re
import numpy as np


class DOMElement:
    """One interactive element from the Gymnasium info dict."""

    def __init__(self, text, x, y, w, h, tag="unknown", ref=""):
        self.text = text
        self.x    = x
        self.y    = y
        self.w    = w
        self.h    = h
        self.tag  = tag
        self.ref  = ref

    def center(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


class DOMFeatureExtractor:
    """
    Parse Gymnasium DOM info → list[DOMElement] → query-match heatmap (H, W).

    Parameters
    ----------
    viewport_width  : int   (matches ScreenProcessor)
    viewport_height : int
    score_threshold : float
    max_elements    : int
    """

    def __init__(self, viewport_width=160, viewport_height=210,
                 score_threshold=0.0, max_elements=200):
        self.vw     = viewport_width
        self.vh     = viewport_height
        self.thresh = score_threshold
        self.max    = max_elements

    def extract_from_info(self, query, info):
        """
        Parameters
        ----------
        query : str   natural-language task description
        info  : dict  from env.step() or env.reset() — expects info["dom_elements"]

        Returns
        -------
        fmap : np.ndarray  (H, W) float32  heatmap
        dom  : list[DOMElement]
        """
        dom  = self._parse(info)
        fmap = self._build_heatmap(query, dom)
        return fmap, dom

    def _parse(self, info):
        raw_elements = info.get("dom_elements", [])
        elements = []
        for i, el in enumerate(raw_elements):
            if i >= self.max:
                break
            try:
                x = float(el.get("left", 0))
                y = float(el.get("top", 0))
                w = float(el.get("width", 0))
                h = float(el.get("height", 0))
                if w == 0 or h == 0:
                    continue
                text = (
                    el.get("text", "")
                    or el.get("value", "")
                    or el.get("id", "")
                    or el.get("classes", "")
                ).strip()[:120]
                tag  = el.get("tag", "unknown").lower()
                ref  = str(el.get("ref", i))
                elements.append(DOMElement(text=text, x=x, y=y, w=w, h=h,
                                           tag=tag, ref=ref))
            except Exception:
                continue
        return elements

    def _build_heatmap(self, query, dom):
        fmap = np.zeros((self.vh, self.vw), dtype=np.float32)
        qw   = _tokenize(query)
        if not qw:
            return fmap
        for el in dom:
            score = _score(qw, el.text)
            if score <= self.thresh:
                continue
            cx, cy = el.center()
            col = int(np.clip(round(cx), 0, self.vw - 1))
            row = int(np.clip(round(cy), 0, self.vh - 1))
            if score > fmap[row, col]:
                fmap[row, col] = score
        return fmap

    def best_match(self, query, dom):
        """Return the DOMElement that best matches query, or None."""
        qw         = _tokenize(query)
        best, best_s = None, 0.0
        for el in dom:
            s = _score(qw, el.text)
            if s > best_s:
                best, best_s = el, s
        return best if best_s > 0.0 else None


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text):
    return [t for t in re.split(r"\W+", text.lower()) if t]


def _score(query_words, text):
    el_words = _tokenize(text)
    if not el_words:
        return 0.0
    return max(_sim(qw, w) for qw in query_words for w in el_words)


def _sim(a, b):
    """Normalised Levenshtein similarity in [0, 1]."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * n
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (ca != cb))
        prev = curr
    return 1.0 - prev[n] / max(m, n)
