"""
env.py

Environment orchestrator — connects all five modules into a single pipeline.

Pipeline
--------
  1. screenshot.py    →  WebScreenshot       captures the page as a 210×160×3 array
  2. image_processing →  screenshot_to_tensor + Global_CNN  encodes the screenshot
  3. DOM_elements.py  →  dom_elements        extracts every DOM node from the page
  4. feature_map.py   →  feature_map         scores each node against the query
  5. local_cnn.py     →  Local_CNN           predicts location + interaction type

Given a URL and a query string, Env runs the full pipeline and prints a
ranked results table showing which elements to interact with, where they are
on screen, and whether to use the cursor or keyboard.

Usage (CLI)
-----------
    python env.py <url> "<query>"

    # example
    python env.py https://example.com "More information"

Usage (Python)
--------------
    from env import Env

    env     = Env()
    results = env.run("https://example.com", "More information")
    for r in results:
        print(r)

Dependencies:
    pip install torch numpy selenium pillow chromedriver-autoinstaller
"""

from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from typing import Optional

import torch

# ── Module imports ─────────────────────────────────────────────────────────────
from screenshot      import WebScreenshot
from image_processing import array_to_tensor, Global_CNN
from DOM_elements    import dom_elements as DOMExtractor, DOMNode
from feature_map     import feature_map  as FeatureMap
from local_cnn       import Local_CNN, ElementPrediction
from local_cnn       import INTERACTION_NONE, INTERACTION_CLICK, INTERACTION_KEYBOARD


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class EnvResult:
    """
    Combined output for a single DOM element after the full pipeline.

    Attributes
    ----------
    rank          : int    – position in the final sorted list (1 = best)
    query_score   : float  – text relevance score from feature_map
    confidence    : float  – Local_CNN confidence
    combined_score: float  – query_score × confidence  (used for final ranking)
    interaction   : str    – 'click', 'keyboard', or 'none'
    predicted_cx  : float  – predicted cursor x in screenshot pixels (0–160)
    predicted_cy  : float  – predicted cursor y in screenshot pixels (0–210)
    tag           : str    – HTML tag name
    text          : str    – visible text of the element (truncated)
    xpath         : str    – XPath that identifies the element
    bbox          : dict   – original bounding box from the browser
    """
    rank:           int
    query_score:    float
    confidence:     float
    combined_score: float
    interaction:    str
    predicted_cx:   float
    predicted_cy:   float
    tag:            str
    text:           str
    xpath:          str
    bbox:           dict

    def __str__(self) -> str:
        text_preview = textwrap.shorten(self.text, width=50, placeholder="…")
        return (
            f"[#{self.rank:>3}] {self.interaction:>8}  "
            f"cx={self.predicted_cx:6.1f}  cy={self.predicted_cy:6.1f}  "
            f"score={self.combined_score:.4f}  "
            f"<{self.tag}>  {text_preview!r}"
        )


# ── Main environment class ────────────────────────────────────────────────────

class Env:
    """
    Connects screenshot → image_processing → DOM_elements → feature_map → local_cnn.

    Parameters
    ----------
    top_k : int
        How many results to return and display. Default 10.
    num_logits : int
        Output size of Global_CNN / Local_CNN head. Default 128.
    wait_seconds : float
        Seconds to wait after page load before capturing. Default 2.
    viewport_width : int
        Browser viewport width. Default 1280.
    viewport_height : int
        Browser viewport height. Default 900.
    verbose : bool
        Print step-by-step progress. Default True.
    """

    def __init__(
        self,
        top_k:           int   = 10,
        num_logits:      int   = 128,
        wait_seconds:    float = 2.0,
        viewport_width:  int   = 1280,
        viewport_height: int   = 900,
        verbose:         bool  = True,
    ):
        self.top_k           = top_k
        self.num_logits      = num_logits
        self.wait_seconds    = wait_seconds
        self.viewport_width  = viewport_width
        self.viewport_height = viewport_height
        self.verbose         = verbose

        # Build models once; they are reused across run() calls
        self._local_cnn: Local_CNN = Local_CNN.build(num_logits=num_logits)
        self._local_cnn.eval()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, url: str, query: str) -> list[EnvResult]:
        """
        Execute the full pipeline for *url* and *query*.

        Returns
        -------
        list[EnvResult]
            Top-k results sorted by combined score (descending).
        """
        self._log(f"\n{'═'*60}")
        self._log(f"  URL   : {url}")
        self._log(f"  Query : {query!r}")
        self._log(f"{'═'*60}")

        # ── Step 1: Screenshot ────────────────────────────────────────────────
        self._log("\n[1/4] Capturing screenshot …")
        ws        = WebScreenshot(
            url,
            wait_seconds   = self.wait_seconds,
            viewport_width = self.viewport_width,
            viewport_height= self.viewport_height,
        )
        rgb_array = ws.capture()
        tensor    = array_to_tensor(rgb_array)          # (1, 3, 210, 160)
        self._log(f"      Array shape : {rgb_array.shape}  dtype={rgb_array.dtype}")

        # ── Step 2: DOM extraction ────────────────────────────────────────────
        self._log("\n[2/4] Extracting DOM elements …")
        extractor = DOMExtractor(
            url,
            wait_seconds   = self.wait_seconds,
            viewport_width = self.viewport_width,
            viewport_height= self.viewport_height,
        )
        nodes = extractor.extract()
        self._log(f"      {len(nodes)} nodes extracted.")

        if not nodes:
            self._log("      No DOM nodes found — aborting.")
            return []

        # ── Step 3: Text feature map ──────────────────────────────────────────
        self._log("\n[3/4] Computing text feature map …")
        fm            = FeatureMap(
            query,
            nodes,
            viewport_width = float(self.viewport_width),
            viewport_height= float(self.viewport_height),
        )
        fm.compute()
        query_scores  = fm.relevance_scores   # np.ndarray (N,)
        self._log(f"      Feature matrix : {fm.matrix.shape}")

        # ── Step 4: Local CNN predictions ─────────────────────────────────────
        self._log("\n[4/4] Running Local_CNN …")
        predictions: list[ElementPrediction] = self._local_cnn.predict(tensor, nodes)
        self._log(f"      {len(predictions)} predictions produced.")

        # ── Merge & rank ──────────────────────────────────────────────────────
        results = self._merge(nodes, query_scores, predictions)
        self._print_results(results, query)
        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _merge(
        self,
        nodes:        list[DOMNode],
        query_scores: "np.ndarray",
        predictions:  list[ElementPrediction],
    ) -> list[EnvResult]:
        """Zip query scores + CNN predictions, compute combined score, sort."""
        merged = []
        for i, (node, pred) in enumerate(zip(nodes, predictions)):
            q_score  = float(query_scores[i])
            conf     = pred.confidence
            combined = q_score * conf          # both factors must be high to rank well

            merged.append(EnvResult(
                rank           = 0,            # filled in after sorting
                query_score    = q_score,
                confidence     = conf,
                combined_score = combined,
                interaction    = pred.interaction_label,
                predicted_cx   = pred.predicted_cx,
                predicted_cy   = pred.predicted_cy,
                tag            = node.tag,
                text           = node.text,
                xpath          = node.xpath,
                bbox           = node.bbox,
            ))

        # Sort descending by combined score
        merged.sort(key=lambda r: r.combined_score, reverse=True)

        # Assign rank and trim to top_k
        for rank, result in enumerate(merged, start=1):
            result.rank = rank
        return merged[: self.top_k]

    def _print_results(self, results: list[EnvResult], query: str):
        if not self.verbose:
            return

        # ── Header ────────────────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  Results for query: {query!r}")
        print(f"{'─'*60}")
        print(
            f"  {'#':>3}  {'action':>8}  "
            f"{'cx':>6}  {'cy':>6}  "
            f"{'score':>8}  {'tag':<10}  text"
        )
        print(f"  {'─'*3}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*30}")

        for r in results:
            text_preview = textwrap.shorten(r.text, width=35, placeholder="…")
            print(
                f"  {r.rank:>3}  {r.interaction:>8}  "
                f"{r.predicted_cx:6.1f}  {r.predicted_cy:6.1f}  "
                f"{r.combined_score:8.4f}  {r.tag:<10}  {text_preview!r}"
            )

        # ── Interaction summary ────────────────────────────────────────────────
        n_click    = sum(1 for r in results if r.interaction == "click")
        n_keyboard = sum(1 for r in results if r.interaction == "keyboard")
        n_none     = sum(1 for r in results if r.interaction == "none")
        print(f"\n  Interaction summary → click: {n_click}  keyboard: {n_keyboard}  none: {n_none}")

        # ── Best match callout ─────────────────────────────────────────────────
        if results:
            best = results[0]
            print(f"\n  ★  Best match")
            print(f"     Tag       : <{best.tag}>")
            print(f"     Text      : {textwrap.shorten(best.text, 60, placeholder='…')!r}")
            print(f"     Action    : {best.interaction}")
            print(f"     Location  : cx={best.predicted_cx:.1f}  cy={best.predicted_cy:.1f}  (screenshot px)")
            print(f"     XPath     : {best.xpath}")
        print(f"{'─'*60}\n")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python env.py <url> \"<query>\"")
        print("       python env.py https://example.com \"More information\"")
        sys.exit(1)

    url   = sys.argv[1]
    query = sys.argv[2]

    env     = Env(top_k=10, verbose=True)
    results = env.run(url, query)

    # Optionally expose results to the caller via exit code 0
    sys.exit(0)


if __name__ == "__main__":
    main()
