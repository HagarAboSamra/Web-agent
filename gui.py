"""
gui.py  —  Web Agent Control Panel
Run:  streamlit run gui.py
"""

import random
import sys
import time
import os

import streamlit as st

st.set_page_config(
    page_title="Web Agent",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("◈ Web Agent")
st.sidebar.caption("DQL  ·  MiniWoB++  ·  CNN Perception")

page = st.sidebar.radio(
    "Navigation",
    ["RUN", "MINIWOB", "TRAIN", "RL THEORY", "ABOUT"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption("Checkpoint")
ckpt_file = st.sidebar.text_input("Path", value="checkpoints/final.npz",
                                  label_visibility="collapsed")
ckpt_exists = os.path.isfile(ckpt_file)
st.sidebar.caption("✓ loaded" if ckpt_exists else "✗ not found")


# ═════════════════════════════════════════════════════════════════════════════
# RUN PAGE  — run a single MiniWoB task with a saved checkpoint
# ═════════════════════════════════════════════════════════════════════════════

if page == "RUN":

    st.title("◈ RUN AGENT")
    st.caption("Load a trained checkpoint and run a MiniWoB++ task.")

    TASKS = [
        "click-button", "click-button-sequence", "click-checkboxes",
        "click-dialog", "click-link", "click-option", "click-tab",
        "click-widget", "enter-text", "enter-password",
        "focus-text", "login-user", "navigate-tree", "search-engine",
        "choose-date", "choose-list", "email-inbox",
    ]

    c1, c2 = st.columns([3, 1])
    task      = c1.selectbox("MiniWoB++ Task", TASKS)
    max_steps = c2.number_input("Max steps", 5, 50, 10)

    if not ckpt_exists:
        st.warning(f"No checkpoint found at `{ckpt_file}`. Train first or point to an existing `.npz`.")

    run_col, _ = st.columns([1, 5])
    run_btn  = run_col.button("▶  Run", use_container_width=True,
                              disabled=not ckpt_exists)

    if run_btn:
        s_col, g_col, d_col, t_col = st.columns(4)
        s_ph = s_col.empty()
        g_ph = g_col.empty()
        d_ph = d_col.empty()
        t_ph = t_col.empty()
        log  = st.empty()

        s_ph.metric("STEPS", "—")
        g_ph.metric("TOTAL RETURN G", "—")
        d_ph.metric("DOM ELEMENTS", "—")
        t_ph.metric("TERMINAL", "—")

        logs = []
        G    = 0.0

        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "env"))
            from environment_handler import ACTIONS
            from miniwob_env         import MiniWoBGymEnv
            import numpy as np

            d    = np.load(ckpt_file)
            W, b = d["W"], d["b"]

            def q_predict(state):
                return state @ W + b

            env = MiniWoBGymEnv(task_name=task, max_steps=max_steps,
                                render_mode=None, seed=0)
            try:
                gen = env.run()
                step_data = next(gen)
                while True:
                    joint    = step_data["joint_vec"]
                    q_vals   = q_predict(joint)
                    ai       = int(q_vals.argmax())
                    act_name = ACTIONS[ai]
                    try:
                        result = gen.send(ai)
                    except StopIteration:
                        break
                    r    = result["miniwob_reward"]
                    done = result["done"]
                    G   += r

                    top3  = q_vals.argsort()[-3:][::-1]
                    q_str = "  ".join(f"{ACTIONS[i]}={q_vals[i]:+.3f}" for i in top3)
                    logs.append(
                        f"[{result['step']:>2}] {act_name:<14}\n"
                        f"      Q-top3: {q_str}\n"
                        f"      r={r:+.3f}  G={G:.3f}"
                    )
                    s_ph.metric("STEPS", result["step"])
                    g_ph.metric("TOTAL RETURN G", f"{G:.3f}")
                    d_ph.metric("DOM ELEMENTS", "—")
                    t_ph.metric("TERMINAL", "✓ YES" if (done and r > 0) else "running…")
                    log.code("\n".join(logs), language="bash")
                    if done:
                        break
                    try:
                        step_data = next(gen)
                    except StopIteration:
                        break
            finally:
                env.close()

            success = G > 0
            t_ph.metric("TERMINAL", "✓ YES" if success else "✗ NO")
            st.success(f"Done  |  G = {G:.4f}  |  {'SUCCESS' if success else 'FAILED'}")

        except Exception as ex:
            st.info(f"Simulation mode ({type(ex).__name__}: {ex})")
            G = 0.0
            for step in range(1, max_steps + 1):
                time.sleep(0.25)
                act = random.choice(["left_click", "scroll_down", "move", "type_text"])
                r   = round(random.uniform(-0.1, 0.5), 3)
                G  += r
                logs.append(f"[{step:>2}] {act}  r={r:+.3f}  G={G:.3f}")
                s_ph.metric("STEPS", step)
                g_ph.metric("TOTAL RETURN G", f"{G:.3f}")
                d_ph.metric("DOM ELEMENTS", random.randint(5, 25))
                log.code("\n".join(logs), language="bash")
            t_ph.metric("TERMINAL", "✓ YES" if G > 0 else "✗ NO")
            st.success(f"Simulation done  |  G = {G:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# MINIWOB PAGE  — train + benchmark on a single task
# ═════════════════════════════════════════════════════════════════════════════

elif page == "MINIWOB":

    st.title("◈ MiniWoB++")
    st.caption("Train and benchmark on controlled browser tasks with ground-truth reward.")

    TASKS = [
        "click-button", "click-button-sequence", "click-checkboxes",
        "click-dialog", "click-link", "click-option", "click-tab",
        "click-widget", "enter-text", "enter-password",
        "focus-text", "login-user", "navigate-tree", "search-engine",
        "choose-date", "choose-list", "email-inbox",
    ]

    c1, c2, c3 = st.columns(3)
    task     = c1.selectbox("Task", TASKS)
    episodes = c2.number_input("Episodes", 10, 500, 50)
    gamma    = c3.number_input("Gamma γ", 0.0, 1.0, 0.99)

    c4, c5 = st.columns(2)
    lr       = c4.number_input("Learning rate", value=1e-3, format="%.4f")
    ckpt_dir = c5.text_input("Checkpoint dir", "checkpoints")

    if st.button("▶  Start", use_container_width=False):

        ep_ph, ret_ph, base_ph, sr_ph, term_ph = st.columns(5)
        log = st.empty()

        logs     = []
        baseline = 0.0
        success  = 0

        ep_ph.metric("EPISODE", "—")
        ret_ph.metric("RETURN G₀", "—")
        base_ph.metric("BASELINE", "—")
        sr_ph.metric("SUCCESS RATE", "—")
        term_ph.metric("TERMINAL", "—")

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "env"))
        sys.path.insert(0, os.path.dirname(__file__))
        from miniwob_env         import MiniWoBGymEnv
        from environment_handler import ACTIONS
        from train import (QNetwork, ReplayBuffer, dql_update,
                           epsilon_greedy, save_checkpoint)
        import numpy as np

        os.makedirs(ckpt_dir, exist_ok=True)

        # probe joint_dim
        _probe = MiniWoBGymEnv(task, max_steps=1, seed=0)
        joint_dim = None
        try:
            for sd in _probe.run():
                joint_dim = sd["joint_vec"].shape[0]
                break
        finally:
            _probe.close()

        if joint_dim is None:
            st.error("Could not determine state dimension. Is `miniwob` installed?  `pip install miniwob`")
            st.stop()

        q_net    = QNetwork(joint_dim, len(ACTIONS), lr=lr)
        q_target = QNetwork(joint_dim, len(ACTIONS), lr=lr)
        q_target.copy_weights_from(q_net)
        replay   = ReplayBuffer(10_000)
        eps      = 1.0
        prev_j   = None

        for ep in range(1, episodes + 1):
            env     = MiniWoBGymEnv(task_name=task, max_steps=15,
                                    render_mode=None, seed=ep)
            total_r  = 0.0
            terminal = False
            try:
                gen = env.run()
                step_data = next(gen)
                while True:
                    joint = step_data["joint_vec"]
                    ai    = epsilon_greedy(q_net, joint, eps, len(ACTIONS))
                    try:
                        result = gen.send(ai)
                    except StopIteration:
                        break
                    r    = result["miniwob_reward"]
                    done = result["done"]
                    if prev_j is not None:
                        replay.push(prev_j, ai, r, joint, float(done))
                    prev_j   = joint
                    total_r += r
                    terminal = result["terminal"]
                    if done:
                        break
                    try:
                        step_data = next(gen)
                    except StopIteration:
                        break
            finally:
                env.close()

            dql_update(q_net, q_target, replay, 32, gamma)
            if ep % 10 == 0:
                q_target.copy_weights_from(q_net)
            eps      = max(0.05, eps * 0.99)
            baseline = 0.95 * baseline + 0.05 * total_r
            if terminal:
                success += 1

            # ── save checkpoint every 100 episodes ──
            if ep % 100 == 0:
                save_checkpoint(q_net, f"{ckpt_dir}/ep_{ep:05d}.npz")

            rate = f"{100 * success // ep}%"
            logs.append(
                f"[{ep:>4}/{episodes}]  G={total_r:+.3f}"
                f"  bl={baseline:+.4f}  ε={eps:.3f}"
                f"  {'✓' if terminal else '✗'}"
            )
            ep_ph.metric("EPISODE", ep)
            ret_ph.metric("RETURN G₀", f"{total_r:.3f}")
            base_ph.metric("BASELINE", f"{baseline:.3f}")
            sr_ph.metric("SUCCESS RATE", rate)
            term_ph.metric("TERMINAL", "✓" if terminal else "✗")
            log.code("\n".join(logs[-30:]), language="bash")

        # ── always save final checkpoint ──
        save_checkpoint(q_net, f"{ckpt_dir}/final.npz")
        st.success(f"Done. Checkpoint saved → {ckpt_dir}/final.npz")


# ═════════════════════════════════════════════════════════════════════════════
# TRAIN PAGE   ← full DQL pipeline: train → checkpoint → run task
# ═════════════════════════════════════════════════════════════════════════════

elif page == "TRAIN":

    st.title("◈ TRAIN  →  RUN")
    st.caption("Train the Q-network on MiniWoB++ tasks, save checkpoint, then run a task with learned weights.")

    st.subheader("Hyperparameters")
    c1, c2, c3, c4 = st.columns(4)
    episodes      = c1.number_input("Episodes",      100, 5000, 300)
    max_steps     = c2.number_input("Max steps",      5,   50,   10)
    lr            = c3.number_input("Learning rate", value=1e-3, format="%.4f")
    gamma         = c4.number_input("Gamma γ",       0.0, 1.0, 0.99)

    c5, c6, c7, c8 = st.columns(4)
    eps_start     = c5.number_input("ε start",   0.0, 1.0, 1.0)
    eps_end       = c6.number_input("ε end",     0.0, 1.0, 0.05)
    eps_decay     = c7.number_input("ε decay",   0.9, 1.0, 0.995)
    target_update = c8.number_input("Target sync every N eps", 1, 50, 10)

    st.subheader("MiniWoB++ Tasks")
    ALL_TASKS = [
        "click-button", "click-link", "focus-text",
        "click-checkboxes", "click-option", "click-tab",
        "enter-text", "click-dialog",
    ]
    miniwob_tasks = st.multiselect(
        "Tasks to train on", ALL_TASKS,
        default=["click-button", "click-link", "focus-text"])

    st.subheader("Run a task after training")
    run_task_name = st.selectbox("Task to test after training", ALL_TASKS)

    ckpt_dir = st.text_input("Checkpoint dir", "checkpoints")

    if st.button("▶  Train + Run", use_container_width=False):

        progress = st.progress(0.0)
        log      = st.empty()
        logs     = []

        st.info("Training…")

        sys.path.insert(0, os.path.dirname(__file__))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "env"))
        from train import (QNetwork, ReplayBuffer, dql_update,
                           epsilon_greedy, save_checkpoint, run_task)
        from environment_handler import ACTIONS
        from miniwob_env         import MiniWoBGymEnv
        import numpy as np, random as rnd

        tasks = miniwob_tasks or ["click-button"]

        os.makedirs(ckpt_dir, exist_ok=True)

        # probe joint_dim
        _probe = MiniWoBGymEnv(tasks[0], max_steps=1, seed=0)
        joint_dim = None
        try:
            for sd in _probe.run():
                joint_dim = sd["joint_vec"].shape[0]
                break
        finally:
            _probe.close()

        if joint_dim is None:
            st.error("Could not determine state dimension. Is `miniwob` installed?  `pip install miniwob`")
            st.stop()

        q_net    = QNetwork(joint_dim, len(ACTIONS), lr=lr)
        q_target = QNetwork(joint_dim, len(ACTIONS), lr=lr)
        q_target.copy_weights_from(q_net)
        replay   = ReplayBuffer(10_000)
        epsilon  = float(eps_start)
        prev_j   = None

        for ep in range(1, episodes + 1):
            task_name = rnd.choice(tasks)
            env       = MiniWoBGymEnv(task_name=task_name,
                                      max_steps=max_steps,
                                      render_mode=None,
                                      seed=ep)
            total_r = 0.0
            ok      = False
            try:
                gen = env.run()
                step_data = next(gen)
                while True:
                    joint = step_data["joint_vec"]
                    ai    = epsilon_greedy(q_net, joint, epsilon, len(ACTIONS))
                    try:
                        result = gen.send(ai)
                    except StopIteration:
                        break
                    r    = result["miniwob_reward"]
                    done = result["done"]
                    if prev_j is not None:
                        replay.push(prev_j, ai, r, joint, float(done))
                    prev_j   = joint
                    total_r += r
                    if r > 0:
                        ok = True
                    if done:
                        break
                    try:
                        step_data = next(gen)
                    except StopIteration:
                        break
            finally:
                env.close()

            dql_update(q_net, q_target, replay, 32, gamma)
            if ep % int(target_update) == 0:
                q_target.copy_weights_from(q_net)
            epsilon = max(float(eps_end), epsilon * float(eps_decay))

            # ── save every 100 episodes ──
            if ep % 100 == 0:
                save_checkpoint(q_net, f"{ckpt_dir}/ep_{ep:05d}.npz")

            logs.append(
                f"Ep {ep:>4}  G={total_r:+.3f}  ε={epsilon:.3f}"
                f"  buf={len(replay)}  {'✓' if ok else '✗'}  [{task_name}]"
            )
            progress.progress(ep / episodes)
            if ep % 5 == 0:
                log.code("\n".join(logs[-25:]), language="bash")

        # ── always save final ──
        save_checkpoint(q_net, f"{ckpt_dir}/final.npz")
        log.code("\n".join(logs[-25:]), language="bash")
        st.success(f"Training complete. Checkpoint saved → {ckpt_dir}/final.npz")

        # ── Run task with trained weights ─────────────────────────────
        st.subheader("Task Execution Result")
        result = run_task(q_net, run_task_name,
                          max_steps=max_steps,
                          render_mode=None, seed=0)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("SUCCESS",      "✓ YES" if result["success"] else "✗ NO")
        r2.metric("TOTAL RETURN", f'{result["total_reward"]:.4f}')
        r3.metric("STEPS",        result["steps"])
        r4.metric("TERMINAL",     "✓ YES" if result["success"] else "✗ NO")
        st.caption(f'Actions: {" → ".join(result["actions_taken"])}')


# ═════════════════════════════════════════════════════════════════════════════
# RL THEORY PAGE  — unchanged
# ═════════════════════════════════════════════════════════════════════════════

elif page == "RL THEORY":

    st.title("◈ RL THEORY")

    st.header("① Markov Decision Process")
    st.markdown(r"""
| Symbol | Meaning |
|--------|---------|
| **S** | screenshot pixels + DOM heatmap + cursor position |
| **A** | {left_click, scroll_up, scroll_down, move, type_text, …} |
| **P** | P(s′ \| s, a) — unknown, sampled by Gymnasium env |
| **R** | MiniWoB++ built-in JavaScript reward |
| **γ** | 0.99 — discount factor |
""")

    st.header("② Bellman Equation")
    st.latex(r"V(s) = R(s,a) + \gamma \cdot V(s')")
    st.latex(r"Q(s,a) = R(s,a) + \gamma \cdot \max_{a'} Q(s', a')")
    st.caption("Q(s,a) is the expected total return from state s, taking action a, then acting optimally.")

    st.header("③ Methods Comparison")
    st.table({
        "Method":   ["Monte Carlo", "TD(0)",    "Q-Learning", "DQL (this project)"],
        "Updates":  ["End of ep",   "Every step","Every step", "Every step"],
        "Variance": ["High",        "Low",       "Low",        "Low"],
        "Bias":     ["None",        "High",      "High",       "High"],
    })

    st.header("④ Deep Q-Learning")
    st.markdown("**Q-network:** linear approximation — `Q(s, a) = s @ W + b`\n\n**ε-greedy selection:**")
    st.latex(r"a = \begin{cases} \text{random} & \text{with prob } \varepsilon \\ \arg\max_a Q(s,a) & \text{otherwise} \end{cases}")
    st.markdown("**Bellman target:**")
    st.latex(r"y = r + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a') \cdot (1 - \text{done})")
    st.markdown("**Loss:**")
    st.latex(r"\mathcal{L} = \bigl(Q(s,a) - y\bigr)^2")
    st.markdown("**Gradient update (W column for action a):**")
    st.latex(r"W_{:,a} \;\leftarrow\; W_{:,a} - \alpha \cdot (Q(s,a) - y) \cdot s")
    st.markdown("""
**Replay buffer:** stores `(s, a, r, s′, done)` tuples, sampled randomly each step
→ breaks temporal correlation, stabilises training.

**Target network:** separate copy of Q-network, synced every C episodes
→ prevents the moving-target problem.
""")

    st.header("⑤ CNN Perception")
    st.markdown("""
```
Gymnasium obs (H×W×3 uint8)
  └─► GlobalCNN     Conv×5 (5×5 / stride 2 / ReLU / He init) → FC(256)
  └─► LocalCNN      Gaussian mask @ cursor → weighted pool → FC(128)
  └─► DOM heatmap   info["dom_elements"] → Levenshtein score → (H×W)

Joint vector = [ global(256) | local(128) | dom(W×H) ]
  └─► grid_head    → WHERE to click   (10×10 grid)
  └─► action_head  → WHAT action      (7 choices)
  └─► key_head     → WHICH key        (72 chars)
```
""")

    st.header("⑥ MiniWoB++")
    st.markdown("""
~100 browser tasks via the Gymnasium API. Each task has:
- A natural-language instruction (`info["utterance"]`)
- Built-in JavaScript reward: **+1.0** success, **−1.0** fail/timeout

| Tier | Examples |
|------|---------|
| Easy | click-button, click-link, focus-text |
| Medium | enter-text, login-user, search-engine |
| Hard | book-flight, buy-ticket, social-media |
""")


# ═════════════════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ═════════════════════════════════════════════════════════════════════════════

elif page == "ABOUT":

    st.title("◈ ABOUT")

    st.markdown("""
## Web Agent — Autonomous Browser Navigation

An RL agent that navigates MiniWoB++ tasks using visual perception and DOM analysis.
Given a task, it decides which actions to take —
clicks, scrolls, keystrokes — using a trained Q-network.

### What makes it work

**Perception pipeline** — every step, the agent:
1. Receives a pixel obs from Gymnasium → GlobalCNN (5 conv layers) → 256-dim vector
2. Applies a Gaussian mask at the cursor → LocalCNN → 128-dim vector
3. Reads `info["dom_elements"]`, scores them → heatmap
4. Concatenates all three → joint state vector

**Decision** — Q-network selects where, what, and which key to press.

**Learning** — DQL with experience replay and a target network.
Episodes accumulate `(s, a, r, s′, done)` transitions; mini-batch updates
minimise the Bellman error.

### Structure

```
gui.py                     ← this file (Streamlit)
train.py                   ← DQL training + task runner
checkpoints/               ← saved Q-network weights (.npz)

env/
  screen_processor.py      Gymnasium obs → (W,H,3) float32
  global_cnn.py            5 conv layers → 256-dim
  local_cnn.py             Gaussian attention → 128-dim
  dom_feature_extractor.py info["dom_elements"] + Levenshtein scoring
  environment_handler.py   ACTIONS, action_to_gym(), SoftmaxHead
  reward.py                Shaped R(s,a) constants (reference)
  miniwob_env.py           MiniWoBGymEnv  (pure Gymnasium wrapper)
```

### Install

```bash
pip install streamlit pillow numpy gymnasium miniwob
```

### Run

```bash
streamlit run gui.py

# CLI training:
python train.py --tasks click-button click-link --episodes 300

# Run only:
python train.py --run-only --resume checkpoints/final.npz --run-task click-button
```
""")
