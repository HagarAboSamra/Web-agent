"""
environment_test.py

CLI test runner for EnvironmentHandler.

Usage
-----
    python environment_test.py <url> [query1] [query2] ...

Examples
--------
    python environment_test.py https://example.com "Click the login button"
    python environment_test.py https://shop.com "Add item to cart" "Proceed to checkout"

The script imports EnvironmentHandler, runs the episode, and prints every
perception–action step in a formatted table — all display logic lives here,
keeping environment_handler.py free of print statements.
"""

import sys
from environment_handler import EnvironmentHandler, PointerEvent, KeyEvent, StepInfo


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

SEP  = "─" * 72
DSEP = "═" * 72


def print_step(info: StepInfo) -> None:
    """Pretty-print one step's StepInfo to stdout."""
    visible = len(info.dom_elements)

    # ── Header ──────────────────────────────────────────────────────────────
    print(
        f"\nSTEP {info.step:<4}  "
        f"visible nodes: {visible:<6}  "
        f"last reward: {info.reward:+.3f}",
        flush=True,
    )
    print(SEP, flush=True)

    # ── Candidate actions table ──────────────────────────────────────────────
    shown = min(visible, 5)
    print(f"\nCandidate actions ({shown} total):", flush=True)
    print(
        f"  {'#':>2}  {'type':<12}  {'real_cx':>7}  {'real_cy':>7}  "
        f"{'conf':>6}  {'tag':<4}  {'text'}",
        flush=True,
    )
    print(
        f"  {'─'*2}  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*4}  {'─'*30}",
        flush=True,
    )

    for i, el in enumerate(info.dom_elements[:5], start=1):
        cx, cy   = el.center
        tag      = getattr(el, "tag",  "?")
        text     = getattr(el, "text", "")
        conf     = getattr(el, "conf", 0.5)
        text_str = repr(text[:30]) if text else "''"
        print(
            f"  {i:>2}  {'CLICK':<12}  {cx:>7.0f}  {cy:>7.0f}  "
            f"{conf:>6.3f}  {tag:<4}  {text_str}",
            flush=True,
        )

    # ── Chosen action detail ─────────────────────────────────────────────────
    print(flush=True)
    action     = info.action
    chosen_el  = info.chosen_el

    if isinstance(action, PointerEvent):
        tx, ty = (chosen_el.center if chosen_el else (action.x, action.y))

        el_text  = chosen_el.text[:40]                           if chosen_el else ""
        el_tag   = chosen_el.tag                                 if chosen_el else "?"
        el_xpath = getattr(chosen_el, "xpath", "N/A")           if chosen_el else "N/A"

        act_label = action.action.upper().replace("_", " ")
        print(
            f"  ► Chosen action : Action({act_label}, ({tx:.0f},{ty:.0f}), "
            f"conf=0.500, target={el_text!r})",
            flush=True,
        )
        print(f"    Target tag     : <{el_tag}>",  flush=True)
        print(f"    Target text    : {el_text!r}", flush=True)
        print(f"    XPath          : {el_xpath}",  flush=True)
        print(f"    Browser coord  : cx={tx:.0f}  cy={ty:.0f}  (CSS px)", flush=True)

        cnn_x = tx / info.W * 256 if info.W else 0
        cnn_y = ty / info.H * 256 if info.H else 0
        print(
            f"    CNN coord      : cx={cnn_x:.1f}  cy={cnn_y:.1f}  "
            f"(screenshot px — training label only)",
            flush=True,
        )

    else:  # KeyEvent
        print(f"  ► Chosen action : KeyEvent(key={action.key!r})", flush=True)

    print(SEP, flush=True)


def print_order_header(order_idx: int, total_orders: int, query: str) -> None:
    print(f"\n{'━'*72}", flush=True)
    print(f"  ORDER {order_idx + 1}/{total_orders}  —  {query!r}", flush=True)
    print(f"{'━'*72}", flush=True)


def print_episode_summary(result) -> None:
    print(f"\n{DSEP}", flush=True)
    status = "✓ All orders succeeded" if result.success else "✗ Episode failed"
    print(f"  Episode result : {status}",        flush=True)
    print(f"  Total steps    : {result.total_steps}", flush=True)
    print(f"  Final URL      : {result.final_url}",   flush=True)
    print(f"  Message        : {result.message}",     flush=True)
    print(f"\n  Per-order summary:", flush=True)
    for o in result.orders:
        sym = "✓" if o.success else "✗"
        print(
            f"    {sym} Order {o.order_index + 1} "
            f"({o.steps_taken} steps): {o.query!r} — {o.message}",
            flush=True,
        )
    print(DSEP, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    url     = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    queries = sys.argv[2:] if len(sys.argv) > 2 else ["Click the more information link"]

    handler = EnvironmentHandler(
        viewport   = "desktop",
        max_steps  = 10,
        headless   = True,
        step_delay = 0.5,
        seed       = 42,
    )

    # ── Print session header ─────────────────────────────────────────────────
    print(handler, flush=True)
    print(f"\nURL    : {url}", flush=True)
    print(f"Orders : {len(queries)}", flush=True)
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}", flush=True)

    # ── Stream steps from the generator ─────────────────────────────────────
    # handler.run() is a generator that yields StepInfo objects and finally
    # returns a RunResult via StopIteration.value.
    gen    = handler.run(url=url, query=queries)
    result = None

    # Track which order we are currently printing a header for
    last_order_step_count = 0
    current_order_idx     = -1

    try:
        while True:
            info: StepInfo = next(gen)

            # Detect order boundary: step counter reset means a new order
            # (global_step resets to 1 at the start of each order loop)
            if info.step <= last_order_step_count or current_order_idx == -1:
                current_order_idx += 1
                print_order_header(current_order_idx, len(queries), queries[current_order_idx])

            last_order_step_count = info.step
            print_step(info)

    except StopIteration as stop:
        result = stop.value

    except Exception as exc:
        print(f"\n  ✗ Episode error: {exc}", flush=True)
        sys.exit(1)

    if result is not None:
        print_episode_summary(result)
        sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()