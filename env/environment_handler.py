"""
environment_handler.py

Master orchestrator that drives a real browser (via Playwright) to complete
a natural-language task on a given URL.

Usage
-----
    from environment_handler import EnvironmentHandler

    handler = EnvironmentHandler(viewport="desktop", max_steps=15)
    result  = handler.run(url="https://example.com", query="Click the login button")

Pipeline per step
-----------------
1.  Screenshot  →  ScreenProcessor         →  normalised tensor
2.  Live DOM    →  DOMFeatureExtractor      →  2-D text feature map + element list
3.  Tensor      →  GlobalCNN               →  global feature vec + intermediate maps
4.  Maps + mouse→  LocalCNN               →  local feature vec
5.  All vecs    →  joint representation    →  three softmax policy heads
6.  Sample      →  PointerEvent | KeyEvent →  Playwright execution
7.  Repeat until task is complete or max_steps reached.

Dependencies
------------
    pip install playwright pillow numpy
    playwright install chromium
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import List

from screen_processor      import ScreenProcessor
from dom_feature_extractor import DOMFeatureExtractor, DOMElement
from global_cnn            import GlobalCNN
from local_cnn             import LocalCNN


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

MOUSE_ACTIONS = ["move", "left_click", "right_click", "double_click",
                 "scroll_up", "scroll_down", "type_text"]

@dataclass
class PointerEvent:
    action: str
    x:      float
    y:      float

    def __post_init__(self) -> None:
        if self.action not in MOUSE_ACTIONS:
            raise ValueError(f"Unknown mouse action '{self.action}'.")

@dataclass
class KeyEvent:
    key:  str   # character or key name, e.g. "Enter", "Tab"
    slot: int = 0


# ---------------------------------------------------------------------------
# Tiny policy head
# ---------------------------------------------------------------------------

class _SoftmaxHead:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 0) -> None:
        rng    = np.random.default_rng(seed)
        std    = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, std, (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.W + self.b
        exp    = np.exp(logits - logits.max())
        return exp / exp.sum()


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class OrderResult:
    """Outcome of a single order within an episode."""
    order_index: int
    query:       str
    success:     bool
    steps_taken: int
    actions:     List[str]
    final_url:   str
    message:     str

@dataclass
class RunResult:
    success:     bool          # True if ALL orders succeeded
    orders:      List["OrderResult"]
    total_steps: int
    final_url:   str
    message:     str


# ---------------------------------------------------------------------------
# Step info container (returned instead of printed)
# ---------------------------------------------------------------------------

@dataclass
class StepInfo:
    """All observable data for one perception–action step."""
    step:         int
    dom_elements: list
    chosen_el:    "DOMElement | None"
    action:       "PointerEvent | KeyEvent"
    reward:       float
    W:            int
    H:            int


# ---------------------------------------------------------------------------
# EnvironmentHandler
# ---------------------------------------------------------------------------

class EnvironmentHandler:
    """
    Drives a Playwright browser to complete a natural-language task.

    Parameters
    ----------
    viewport : str | tuple[int, int]
        Browser viewport.  One of ``"desktop"``, ``"tablet"``, ``"mobile"``
        or a custom ``(width, height)`` tuple.
    max_steps : int
        Hard limit on the number of perception–action iterations.
    grid_rows, grid_cols : int
        Coarseness of the mouse-movement grid used by the spatial head.
    key_vocab : list[str] | None
        Typeable characters / key names for the keyboard head.
    headless : bool
        Run Chrome headlessly (no visible window).
    step_delay : float
        Seconds to wait after each browser action (lets the page settle).
    seed : int
        Master RNG seed for all sub-components.
    """

    def __init__(
        self,
        viewport:   str | tuple = "desktop",
        max_steps:  int   = 20,
        grid_rows:  int   = 10,
        grid_cols:  int   = 10,
        key_vocab:  List[str] | None = None,
        headless:   bool  = True,
        step_delay: float = 0.8,
        seed:       int   = 0,
    ) -> None:
        self.viewport   = viewport
        self.max_steps  = max_steps
        self.grid_rows  = grid_rows
        self.grid_cols  = grid_cols
        self.key_vocab  = key_vocab or list(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789 .,!?@-_"
        )
        self.headless   = headless
        self.step_delay = step_delay

        # Sub-components
        self.screen_proc = ScreenProcessor(viewport=viewport)
        W = self.screen_proc.expected_width
        H = self.screen_proc.expected_height

        self.dom_extractor = DOMFeatureExtractor(
            viewport_width  = W,
            viewport_height = H,
        )
        self.global_cnn = GlobalCNN(viewport=viewport, seed=seed)
        self.local_cnn  = LocalCNN(seed=seed + 1)

        # Joint dimension
        global_dim = self.global_cnn.fc_out
        local_dim  = self.local_cnn.output_dim
        dom_flat   = W * H
        joint_dim  = global_dim + local_dim + dom_flat

        # Policy heads
        self._grid_head   = _SoftmaxHead(joint_dim, grid_rows * grid_cols, seed + 2)
        self._action_head = _SoftmaxHead(joint_dim, len(MOUSE_ACTIONS),    seed + 3)
        self._key_head    = _SoftmaxHead(joint_dim, len(self.key_vocab),   seed + 4)

        self._W = W
        self._H = H

    # ------------------------------------------------------------------
    # Main public interface
    # ------------------------------------------------------------------

    def run(self, url: str, query: str | List[str]) -> RunResult:
        """
        Navigate to ``url`` and attempt to complete one or more orders in a
        single browser episode.

        Parameters
        ----------
        url   : str               – start URL, e.g. ``"https://example.com"``
        query : str | list[str]   – one task string, or a list of tasks to
                                    execute sequentially within the same
                                    browser session.

        Returns
        -------
        RunResult  – episode summary; per-order details in ``result.orders``.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ImportError(
                "Playwright is required: pip install playwright && "
                "playwright install chromium"
            ) from exc

        queries: List[str] = [query] if isinstance(query, str) else list(query)
        order_results: List[OrderResult] = []
        total_steps = 0

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=self.headless)
            page    = browser.new_page(
                viewport=self.screen_proc.playwright_viewport()
            )

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                time.sleep(self.step_delay)

                # Track cursor position across the whole episode
                mouse_x, mouse_y = float(self._W // 2), float(self._H // 2)

                for order_idx, current_query in enumerate(queries):
                    action_log: List[str] = []
                    order_start_url = page.url

                    order_success = False
                    order_msg     = f"Max steps ({self.max_steps}) reached."

                    for step in range(self.max_steps):
                        global_step = total_steps + step + 1

                        # 1. Capture current state
                        png_bytes = page.screenshot()
                        dom_map, dom_elements = self.dom_extractor.extract_from_page(
                            current_query, page
                        )

                        # 2. Perceive
                        norm_pixels = self.screen_proc.process(png_bytes)
                        global_feats, interm_maps = self.global_cnn.forward(norm_pixels)
                        local_feats = self.local_cnn.forward(
                            interm_maps, mouse_x, mouse_y, self._W, self._H
                        )

                        # 3. Build joint representation
                        joint = np.concatenate([
                            global_feats,
                            local_feats,
                            dom_map.ravel(),
                        ])

                        # 4. Decode action distributions
                        grid_probs   = self._grid_head.forward(joint)
                        action_probs = self._action_head.forward(joint)
                        key_probs    = self._key_head.forward(joint)

                        # 5. Pick best-matching DOM element for targeting
                        best_element = self._best_dom_match(current_query, dom_elements)

                        # 6. Decode & execute the action
                        action  = self._decode_action(
                            grid_probs, action_probs, key_probs, current_query
                        )
                        log_str = self._execute(action, page, best_element, mouse_x, mouse_y)
                        action_log.append(f"Step {global_step}: {log_str}")

                        # 7. Yield step info (caller decides how to display it)
                        yield StepInfo(
                            step         = global_step,
                            dom_elements = dom_elements,
                            chosen_el    = best_element,
                            action       = action,
                            reward       = 0.0,
                            W            = self._W,
                            H            = self._H,
                        )

                        # Update tracked cursor position
                        if isinstance(action, PointerEvent):
                            mouse_x, mouse_y = action.x, action.y

                        time.sleep(self.step_delay)

                        # 8. Check termination for THIS order only
                        if self._task_complete(page, current_query, order_start_url):
                            order_success = True
                            order_msg     = "Order completed successfully."
                            total_steps  += step + 1
                            break

                    else:
                        total_steps += self.max_steps

                    order_results.append(OrderResult(
                        order_index = order_idx,
                        query       = current_query,
                        success     = order_success,
                        steps_taken = len(action_log),
                        actions     = action_log,
                        final_url   = page.url,
                        message     = order_msg,
                    ))

                    if not order_success:
                        break

                all_ok = all(o.success for o in order_results)
                return RunResult(
                    success     = all_ok,
                    orders      = order_results,
                    total_steps = total_steps,
                    final_url   = page.url,
                    message     = (
                        f"All {len(queries)} orders completed."
                        if all_ok
                        else f"Episode stopped after order {len(order_results)}."
                    ),
                )

            finally:
                browser.close()

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------

    def _decode_action(
        self,
        grid_probs:   np.ndarray,
        action_probs: np.ndarray,
        key_probs:    np.ndarray,
        query:        str,
    ) -> PointerEvent | KeyEvent:
        """
        Sample from the policy heads and return a concrete action.
        """
        rng = np.random.default_rng()   # unseeded → different each call

        action_idx  = int(rng.choice(len(MOUSE_ACTIONS), p=action_probs))
        action_name = MOUSE_ACTIONS[action_idx]

        if action_name == "type_text":
            key_idx = int(rng.choice(len(self.key_vocab), p=key_probs))
            return KeyEvent(key=self.key_vocab[key_idx], slot=key_idx)

        grid_idx = int(rng.choice(len(grid_probs), p=grid_probs))
        row      = grid_idx // self.grid_cols
        col      = grid_idx  % self.grid_cols

        px = (col + 0.5) * (self._W / self.grid_cols)
        py = (row + 0.5) * (self._H / self.grid_rows)

        return PointerEvent(action=action_name, x=float(px), y=float(py))

    # ------------------------------------------------------------------
    # Playwright action execution
    # ------------------------------------------------------------------

    def _execute(
        self,
        action:       PointerEvent | KeyEvent,
        page,
        best_element: DOMElement | None,
        mouse_x:      float,
        mouse_y:      float,
    ) -> str:
        """
        Translate an action object into a Playwright call and return a
        human-readable log string describing what was executed.
        """
        if isinstance(action, KeyEvent):
            page.keyboard.press(action.key) if len(action.key) > 1 \
                else page.keyboard.type(action.key)
            return f"KeyEvent(key={action.key!r})"

        tx, ty = action.x, action.y
        if best_element and action.action in ("left_click", "double_click", "right_click"):
            tx, ty = best_element.center

        try:
            if action.action == "move":
                page.mouse.move(tx, ty)
            elif action.action == "left_click":
                page.mouse.click(tx, ty)
            elif action.action == "right_click":
                page.mouse.click(tx, ty, button="right")
            elif action.action == "double_click":
                page.mouse.dblclick(tx, ty)
            elif action.action == "scroll_up":
                page.mouse.wheel(0, -300)
            elif action.action == "scroll_down":
                page.mouse.wheel(0, 300)
        except Exception as exc:
            return f"PointerEvent({action.action} @ {tx:.0f},{ty:.0f}) — ERROR: {exc}"

        elem_label = f" [{best_element.text[:30]}]" if best_element else ""
        return f"PointerEvent({action.action} @ {tx:.0f},{ty:.0f}){elem_label}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _best_dom_match(
        self,
        query: str,
        dom:   List[DOMElement],
    ) -> DOMElement | None:
        """Return the DOM element whose text best matches the query."""
        if not dom:
            return None
        query_words = self.dom_extractor._tokenize(query)
        scored = [
            (self.dom_extractor._element_score(query_words, el.text), el)
            for el in dom
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        best_score, best_el = scored[0]
        return best_el if best_score > 0.3 else None

    @staticmethod
    def _task_complete(page, query: str, start_url: str) -> bool:
        """
        Heuristic termination check.
        Returns True if the page URL changed or a common success indicator
        is found in the page title / URL.
        """
        current_url = page.url
        if current_url != start_url:
            return True
        title = (page.title() or "").lower()
        success_keywords = ["success", "confirmed", "thank you", "welcome",
                            "dashboard", "home", "signed in", "logged in"]
        return any(kw in title for kw in success_keywords)

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EnvironmentHandler("
            f"viewport={self.viewport!r}, "
            f"max_steps={self.max_steps}, "
            f"grid={self.grid_rows}×{self.grid_cols}, "
            f"headless={self.headless})"
        )