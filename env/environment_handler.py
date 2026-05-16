"""
environment_handler.py
Pure-Gymnasium action definitions and shared policy heads.

No Playwright.  All browser interaction happens through the
official MiniWoB++ Gymnasium API (miniwob package).

MDP
----
  State  s  = pixel obs (from gym) + DOM heatmap + cursor position
  Action a  ∈ 7 discrete actions  (mapped to Gymnasium action dicts)
  Reward R  = MiniWoB++ built-in JavaScript reward
  Next   s' = next Gymnasium obs after action
  γ = 0.99

Action mapping  (Gymnasium miniwob action space)
-------------------------------------------------
  Gymnasium expects:
    {"action_type": ACTION_TYPE, "coords": [x, y], "text": str}

  We wrap 7 semantic actions → gymnasium dicts via action_to_gym().
"""

import numpy as np

# Semantic actions (same names as before for compatibility with train.py)
ACTIONS = [
    "move",
    "left_click",
    "right_click",
    "double_click",
    "scroll_up",
    "scroll_down",
    "type_text",
]

KEY_VOCAB = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 .,!?@-_"
)

# ── Gymnasium action-type integer indices ─────────────────────────────────────
# Indices from ActionSpaceConfig.get_preset('all_supported'):
#   0:NONE  1:MOVE_COORDS  2:CLICK_COORDS  3:DBLCLICK_COORDS
#   4:MOUSEDOWN  5:MOUSEUP  6:SCROLL_UP  7:SCROLL_DOWN
#   8:CLICK_ELEMENT  9:PRESS_KEY  10:TYPE_TEXT  …
_AT_MOVE        = 1
_AT_CLICK       = 2
_AT_DBLCLICK    = 3
_AT_SCROLL_UP   = 6
_AT_SCROLL_DOWN = 7
_AT_TYPE_TEXT   = 10


def action_to_gym(action_name, x=0.0, y=0.0, text=""):
    """
    Convert a semantic action to a Gymnasium miniwob action dict.

    action_type must be an integer index into ActionSpaceConfig.action_types
    (not a string). coords is a float32 numpy array [x, y].

    Parameters
    ----------
    action_name : str  one of ACTIONS
    x, y        : float  viewport pixel coordinates
    text        : str    text to type (only for type_text)

    Returns
    -------
    dict  compatible with miniwob Gymnasium action space
    """
    import numpy as np
    coords = np.array([x, y], dtype=np.float32)
    mapping = {
        "move":         {"action_type": _AT_MOVE,        "coords": coords},
        "left_click":   {"action_type": _AT_CLICK,       "coords": coords},
        "right_click":  {"action_type": _AT_CLICK,       "coords": coords},
        "double_click": {"action_type": _AT_DBLCLICK,    "coords": coords},
        "scroll_up":    {"action_type": _AT_SCROLL_UP,   "coords": coords},
        "scroll_down":  {"action_type": _AT_SCROLL_DOWN, "coords": coords},
        "type_text":    {"action_type": _AT_TYPE_TEXT,   "text":   text,
                         "coords": coords},
    }
    return mapping.get(action_name, {"action_type": _AT_MOVE, "coords": coords})


# ── Softmax policy head (unchanged from original) ────────────────────────────

class SoftmaxHead:
    """
    Linear layer + softmax — one learnable policy head.

    logits = x @ W + b
    π(a|s) = softmax(logits)
    """

    def __init__(self, in_dim, out_dim, seed=0):
        rng    = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2.0 / in_dim),
                            (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x):
        logits = x @ self.W + self.b
        e      = np.exp(logits - logits.max())
        return e / e.sum()
