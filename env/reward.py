"""
reward.py
Shaped reward function for every environment step.

Reward Table
------------
  +1.0   URL changed            → navigated to a new page
  +0.6   page title matches query keyword
  +0.4   clicked element matches query  (scaled by score)
  +0.2   input field focused    → ready to type
  +0.1   scroll made progress
  -0.05  step penalty           → encourages efficiency
  -0.1   same action repeated ≥3 times (stuck)
  -1.0   episode timed out

All values clipped to [-1, +1].

MDP
----
  State  s  = screenshot pixels + DOM heatmap + cursor position
  Action a  ∈ {move, left_click, right_click, double_click,
               scroll_up, scroll_down, type_text}
  Reward R  = shaped signal below
  Next   s' = new screenshot after action
  γ = 0.99

Bellman / Q-Learning link
--------------------------
  Q(s,a) = R(s,a) + γ · max_a' Q(s',a')

  DQL update (train.py):
    target = r + γ · max_a' Q_target(s',a')
    loss   = ( Q(s,a) − target )²
"""

URL_CHANGE   = +1.0
TITLE_MATCH  = +0.6
DOM_MATCH    = +0.4
INPUT_FOCUS  = +0.2
SCROLL_PROG  = +0.1
STEP_PENALTY = -0.05
REPEAT_PEN   = -0.1
TIMEOUT      = -1.0
REPEAT_N     = 3


class StepState:
    """Snapshot of one environment step, passed to compute()."""

    def __init__(self, step_index, prev_url, curr_url, page_title, query,
                 action_name, dom_match_score, focused_tag,
                 scroll_y_before, scroll_y_after, timed_out,
                 action_history=None):
        self.step_index      = step_index
        self.prev_url        = prev_url
        self.curr_url        = curr_url
        self.page_title      = page_title
        self.query           = query
        self.action_name     = action_name
        self.dom_match_score = dom_match_score
        self.focused_tag     = focused_tag
        self.scroll_y_before = scroll_y_before
        self.scroll_y_after  = scroll_y_after
        self.timed_out       = timed_out
        self.action_history  = action_history or []


def compute(state):
    """
    Return shaped reward for one step, clipped to [-1, +1].

    Parameters
    ----------
    state : StepState

    Returns
    -------
    float
    """
    if state.timed_out:
        return -1.0

    r = STEP_PENALTY

    if state.curr_url != state.prev_url:
        r += URL_CHANGE

    title = state.page_title.lower()
    if any(w in title for w in state.query.lower().split()):
        r += TITLE_MATCH

    if state.dom_match_score > 0.5:
        r += DOM_MATCH * state.dom_match_score

    if state.focused_tag in ("input", "textarea", "select"):
        r += INPUT_FOCUS

    if state.action_name in ("scroll_down", "scroll_up"):
        if abs(state.scroll_y_after - state.scroll_y_before) > 10:
            r += SCROLL_PROG

    if (len(state.action_history) >= REPEAT_N
            and len(set(state.action_history[-REPEAT_N:])) == 1):
        r += REPEAT_PEN

    return float(max(-1.0, min(1.0, r)))
