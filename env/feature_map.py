"""
feature_map.py

Computes a text-based feature map by matching a query string against
the DOM elements extracted by dom_elements.py.

The feature map is a 2-D float32 array of shape (N, F) where:
  N = number of DOM nodes
  F = number of text features per node

Features per node
-----------------
  0  exact_match        1.0 if query == node text (case-insensitive)
  1  contains_match     1.0 if query is a substring of node text
  2  token_overlap      Jaccard similarity of query tokens vs node text tokens
  3  char_similarity    Normalised character-level edit distance (1 - edit/max)
  4  tag_weight         Heuristic importance weight by HTML tag
  5  is_clickable       1.0 / 0.0
  6  is_typeable        1.0 / 0.0
  7  is_visible         1.0 / 0.0
  8  attr_match         1.0 if any attribute VALUE contains the query
  9  normalised_cx      Element centre-x normalised to [0, 1] by viewport width
  10 normalised_cy      Element centre-y normalised to [0, 1] by viewport height

Dependencies:
    pip install numpy
    (No external NLP libraries required – pure Python + NumPy)
"""

from __future__ import annotations

import re
import numpy as np
from typing import Optional

# Import the DOMNode dataclass from DOM_elements
from DOM_elements import DOMNode


# ── Constants ──────────────────────────────────────────────────────────────────

NUM_FEATURES = 11  # columns in the feature matrix

# Higher weight → more likely to be the target of a query
_TAG_WEIGHTS: dict[str, float] = {
    "a":        1.0,
    "button":   1.0,
    "input":    0.9,
    "textarea": 0.9,
    "select":   0.8,
    "label":    0.7,
    "h1":       0.8,
    "h2":       0.7,
    "h3":       0.6,
    "h4":       0.5,
    "li":       0.4,
    "td":       0.4,
    "th":       0.5,
    "span":     0.3,
    "p":        0.3,
    "div":      0.2,
}


# ── Helper functions ───────────────────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    """Lower-case, strip punctuation, split on whitespace."""
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return intersection / union if union else 0.0


def _edit_distance(s1: str, s2: str) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    m, n = len(s1), len(s2)
    if m < n:
        s1, s2, m, n = s2, s1, n, m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if s1[i - 1] == s2[j - 1] else 1),
            )
        prev = curr
    return prev[n]


def _char_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - _edit_distance(a[:50], b[:50]) / max(len(a[:50]), len(b[:50]), 1)


# ── Main class ────────────────────────────────────────────────────────────────

class feature_map:
    """
    Computes a text feature map representing the relevance of each DOM node
    to a given query string.

    Parameters
    ----------
    query : str
        The search or instruction text (e.g. "click Sign in").
    dom_nodes : list[DOMNode]
        DOM elements extracted by ``dom_elements.extract()``.
    viewport_width : float
        Used to normalise cx coordinates. Default 1280.
    viewport_height : float
        Used to normalise cy coordinates. Default 900.

    Usage
    -----
    >>> fm = feature_map("search", nodes)
    >>> matrix = fm.compute()          # shape (N, 11)
    >>> scores  = fm.relevance_scores  # shape (N,)  – composite score per node
    >>> top5    = fm.top_k(5)          # [(score, DOMNode), ...]
    """

    def __init__(
        self,
        query: str,
        dom_nodes: list[DOMNode],
        viewport_width: float  = 1280.0,
        viewport_height: float = 900.0,
    ):
        self.query           = query.strip()
        self.dom_nodes       = dom_nodes
        self.viewport_width  = viewport_width
        self.viewport_height = viewport_height

        self._matrix: Optional[np.ndarray]  = None   # (N, NUM_FEATURES)
        self._scores: Optional[np.ndarray]  = None   # (N,)

        # Pre-tokenise query once
        self._q_lower  = self.query.lower()
        self._q_tokens = _tokenise(self.query)

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(self) -> np.ndarray:
        """
        Build and return the feature matrix of shape (N, NUM_FEATURES).
        Also populates ``relevance_scores``.
        """
        N = len(self.dom_nodes)
        matrix = np.zeros((N, NUM_FEATURES), dtype=np.float32)

        for i, node in enumerate(self.dom_nodes):
            matrix[i] = self._node_features(node)

        self._matrix = matrix
        # Composite score: weighted sum of key features
        weights = np.array(
            [3.0, 2.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.3, 1.0, 0.0, 0.0],
            dtype=np.float32,
        )
        self._scores = matrix @ weights
        return matrix

    @property
    def matrix(self) -> Optional[np.ndarray]:
        """Feature matrix (N, NUM_FEATURES), or None before compute()."""
        return self._matrix

    @property
    def relevance_scores(self) -> Optional[np.ndarray]:
        """Composite relevance score per node (N,), or None before compute()."""
        return self._scores

    def top_k(self, k: int = 5) -> list[tuple[float, DOMNode]]:
        """
        Return the top-k (score, DOMNode) pairs in descending score order.
        Calls ``compute()`` if not yet done.
        """
        if self._scores is None:
            self.compute()
        k = min(k, len(self.dom_nodes))
        idx = np.argsort(self._scores)[::-1][:k]
        return [(float(self._scores[i]), self.dom_nodes[i]) for i in idx]

    def best_match(self) -> Optional[tuple[float, DOMNode]]:
        """Return the single best-matching (score, DOMNode) pair."""
        results = self.top_k(1)
        return results[0] if results else None

    def feature_names(self) -> list[str]:
        return [
            "exact_match",
            "contains_match",
            "token_overlap",
            "char_similarity",
            "tag_weight",
            "is_clickable",
            "is_typeable",
            "is_visible",
            "attr_match",
            "normalised_cx",
            "normalised_cy",
        ]

    # ── Private helpers ────────────────────────────────────────────────────────

    def _node_features(self, node: DOMNode) -> np.ndarray:
        feat = np.zeros(NUM_FEATURES, dtype=np.float32)
        node_text   = node.text.lower()
        node_tokens = _tokenise(node.text)

        # 0  exact_match
        feat[0] = 1.0 if self._q_lower == node_text else 0.0

        # 1  contains_match
        feat[1] = 1.0 if self._q_lower and self._q_lower in node_text else 0.0

        # 2  token_overlap (Jaccard)
        feat[2] = _jaccard(self._q_tokens, node_tokens)

        # 3  char_similarity
        feat[3] = _char_similarity(self._q_lower, node_text[:80])

        # 4  tag_weight
        feat[4] = _TAG_WEIGHTS.get(node.tag, 0.1)

        # 5  is_clickable
        feat[5] = 1.0 if node.is_clickable else 0.0

        # 6  is_typeable
        feat[6] = 1.0 if node.is_typeable else 0.0

        # 7  is_visible
        feat[7] = 1.0 if node.is_visible else 0.0

        # 8  attr_match – does any attribute value contain the query?
        feat[8] = float(
            any(self._q_lower in str(v).lower() for v in node.attributes.values())
        )

        # 9  normalised_cx
        feat[9] = node.cx / self.viewport_width if self.viewport_width else 0.0

        # 10 normalised_cy
        feat[10] = node.cy / self.viewport_height if self.viewport_height else 0.0

        return feat


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from DOM_elements import dom_elements as DOMExtractor
    import sys

    url   = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    query = sys.argv[2] if len(sys.argv) > 2 else "More information"

    print(f"[feature_map] URL   : {url}")
    print(f"[feature_map] Query : {query!r}")

    extractor = DOMExtractor(url)
    nodes     = extractor.extract()

    fm     = feature_map(query, nodes)
    matrix = fm.compute()

    print(f"\nFeature matrix shape : {matrix.shape}")
    print(f"Feature names        : {fm.feature_names()}")

    print("\n=== Top 5 matches ===")
    for score, node in fm.top_k(5):
        print(f"  score={score:.3f}  {node}")
