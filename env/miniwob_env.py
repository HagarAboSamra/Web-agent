"""
miniwob_env.py
MiniWoB++ Gymnasium environment wrapper — no Playwright, no browser automation.

Uses the official Farama Foundation `miniwob` package:
    pip install miniwob

Quick Start
-----------
    import gymnasium as gym
    import miniwob                            # registers envs

    env = MiniWoBGymEnv("click-button")
    for step in env.run():
        print(step)

MiniWoB++ Gymnasium API
-----------------------
  obs, info = env.reset()
  obs       → np.ndarray  (H, W, 3) uint8  RGB pixel observation
  info      → dict  with "utterance" (task string) and "dom_elements"

  obs, reward, terminated, truncated, info = env.step(action)
  action    → dict  {"action_type": ..., "coords": [x,y], "text": ...}

Reference: https://miniwob.farama.org/
"""

import numpy as np

from screen_processor      import ScreenProcessor
from dom_feature_extractor import DOMFeatureExtractor
from global_cnn            import GlobalCNN
from local_cnn             import LocalCNN
from environment_handler   import (
    ACTIONS, KEY_VOCAB, SoftmaxHead, action_to_gym
)

EASY_TASKS = [
    "click-button", "click-link", "click-checkboxes",
    "focus-text", "click-option", "click-tab", "click-widget",
]
MEDIUM_TASKS = [
    "enter-text", "enter-password", "click-dialog",
    "navigate-tree", "search-engine", "choose-date",
]
HARD_TASKS = [
    "book-flight", "login-user", "buy-ticket", "terminal",
]
ALL_TASKS = EASY_TASKS + MEDIUM_TASKS + HARD_TASKS


class MiniWoBGymEnv:
    """
    Thin wrapper around the official MiniWoB++ Gymnasium env.

    Handles:
      - env creation via gymnasium.make("miniwob/{task_name}-v1")
      - perception pipeline: pixels → GlobalCNN → LocalCNN → joint vector
      - DOM heatmap from info["dom_elements"]
      - action execution via action_to_gym()

    Parameters
    ----------
    task_name   : str   e.g. "click-button"
    max_steps   : int
    render_mode : str | None   None = headless (default) or "human" (visible browser)
    seed        : int
    """

    def __init__(self, task_name="click-button", max_steps=10,
                 render_mode=None, seed=0):
        self.task_name   = task_name
        self.max_steps   = max_steps
        self.render_mode = render_mode
        self.seed        = seed

        # Build perception pipeline (native MiniWoB resolution 160×210)
        self.screen_proc   = ScreenProcessor("miniwob")
        self._W            = self.screen_proc.w    # 160
        self._H            = self.screen_proc.h    # 210
        self.dom_extractor = DOMFeatureExtractor(self._W, self._H)
        self.global_cnn    = GlobalCNN(viewport=(self._W, self._H), seed=seed)
        self.local_cnn     = LocalCNN(seed=seed + 1)

        # MiniWoB++ task iframe is 160×160 (the 210 px total includes a 50 px
        # query header that is NOT part of the selenium-clickable area).
        # Selenium coords MUST stay within [0, _TASK_W) × [0, _TASK_H).
        self._TASK_W = self._W   # 160
        self._TASK_H = 160       # task area only — NOT self._H (210)

        joint_dim          = (self.global_cnn.fc_out
                              + self.local_cnn.output_dim
                              + self._W * self._H)
        self._grid_head    = SoftmaxHead(joint_dim, 100,            seed + 2)
        self._action_head  = SoftmaxHead(joint_dim, len(ACTIONS),   seed + 3)
        self._key_head     = SoftmaxHead(joint_dim, len(KEY_VOCAB),  seed + 4)

        self._gym_env      = None

    # ── Gymnasium env lifecycle ───────────────────────────────────────────────

    def _make_env(self):
        import gymnasium as gym
        import miniwob  # noqa: F401 — registers environments
        env = gym.make(
            f"miniwob/{self.task_name}-v1",
            render_mode=self.render_mode,
        )
        return env

    def close(self):
        if self._gym_env is not None:
            self._gym_env.close()
            self._gym_env = None

    # ── Episode runner ────────────────────────────────────────────────────────

    def run(self):
        """
        Run one episode.  Supports external action injection via .send(ai).

        Usage (internal policy):
            for step_data in env.run():
                ...

        Usage (external policy — Q-network / epsilon-greedy):
            gen = env.run()
            step_data = next(gen)          # first step: perceive only
            while True:
                ai = q_net.predict(step_data["joint_vec"]).argmax()
                try:
                    step_data = gen.send(ai)   # execute ai, get next perception
                except StopIteration:
                    break

        Yields dict per step:
            step, joint_vec, action_idx, grid_idx, key_idx,
            miniwob_reward, done, terminal

        StopIteration value:
            dict  {success, total_reward, steps}
        """
        if self._gym_env is None:
            self._gym_env = self._make_env()

        obs, info = self._gym_env.reset(seed=self.seed)
        query     = info.get("utterance", "complete the task")

        mx, my    = float(self._TASK_W / 2), float(self._TASK_H / 2)
        total_r   = 0.0
        success   = False
        rng       = np.random.default_rng(self.seed)

        for step in range(self.max_steps):
            # ── Perceive ──────────────────────────────────────────────────
            pixels       = self.screen_proc.process(obs)
            dom_map, els = self.dom_extractor.extract_from_info(query, info)
            g_feat, maps = self.global_cnn.forward(pixels)
            l_feat       = self.local_cnn.forward(maps, mx, my, self._TASK_W, self._TASK_H)
            joint        = np.concatenate([g_feat, l_feat, dom_map.ravel()])

            # ── Yield perception; accept external action via .send(ai) ────
            grid_p   = self._grid_head.forward(joint)
            action_p = self._action_head.forward(joint)
            key_p    = self._key_head.forward(joint)

            # Yield joint_vec first; caller may .send(ai) to override action
            external_ai = yield {
                "step":           step + 1,
                "joint_vec":      joint,
                "action_idx":     None,   # filled after action is chosen
                "grid_idx":       None,
                "key_idx":        None,
                "miniwob_reward": 0.0,    # filled after env.step()
                "done":           False,
                "terminal":       False,
            }

            # Use external action if provided, else fall back to internal policy
            if external_ai is not None:
                ai = int(external_ai)
            else:
                ai = int(rng.choice(len(ACTIONS), p=action_p))
            act_name = ACTIONS[ai]

            if act_name == "type_text":
                ki      = int(rng.choice(len(KEY_VOCAB), p=key_p))
                gi      = 0
                key_idx = ki
                k       = KEY_VOCAB[ki]
                gym_action = action_to_gym("type_text", text=k)
            else:
                gi      = int(rng.choice(len(grid_p), p=grid_p))
                key_idx = -1
                row, col = gi // 10, gi % 10
                # Use TASK dimensions (160×160) — NOT self._H (210) which
                # includes the 50 px query header.  Rows 8-9 would give
                # y > 160 with self._H, causing MoveTargetOutOfBoundsException.
                px = (col + 0.5) * (self._TASK_W / 10)
                py = (row + 0.5) * (self._TASK_H / 10)

                # DOM-guided override for click actions
                best_el = self.dom_extractor.best_match(query, els)
                if best_el and act_name in ("left_click", "double_click", "right_click"):
                    bx, by = best_el.center()
                    # DOM coords are relative to task area; clamp defensively.
                    px = float(np.clip(bx, 0.0, self._TASK_W - 1.0))
                    py = float(np.clip(by, 0.0, self._TASK_H - 1.0))

                # Hard safety clamp — always keep within task iframe bounds.
                px = float(np.clip(px, 0.0, self._TASK_W - 1.0))
                py = float(np.clip(py, 0.0, self._TASK_H - 1.0))

                gym_action = action_to_gym(act_name, x=px, y=py)
                mx, my = px, py

            # ── Step ──────────────────────────────────────────────────────
            obs, reward, terminated, truncated, info = self._gym_env.step(gym_action)
            done     = terminated or truncated or step == self.max_steps - 1
            total_r += reward
            if reward > 0:
                success = True

            # Yield the result of the executed action
            yield {
                "step":           step + 1,
                "joint_vec":      joint,
                "action_idx":     ai,
                "grid_idx":       gi,
                "key_idx":        key_idx,
                "miniwob_reward": reward,
                "done":           done,
                "terminal":       done,
            }

            if done:
                break

        return {"success": success, "total_reward": total_r, "steps": step + 1}
