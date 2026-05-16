# ◈ Web Agent — Autonomous Browser Navigation

> RL agent that navigates real websites using visual perception, DOM analysis,
> and **Deep Q-Learning** — no labelled data, no hardcoded rules.

---

## What it does

Give it a URL and a natural-language task. It takes a screenshot, reads the DOM,
and decides: **click / scroll / type**. After training, it runs the task greedily
and prints each step with Q-values and rewards.

---

## Architecture

```
Screenshot (PNG)
     │
     ▼
ScreenProcessor ──► (W, H, 3) float32
     │
     ├──► GlobalCNN ──► 256-dim
     │      Conv×5  (5×5 / stride 2 / ReLU / He init)
     │      FC → 256
     │
     ├──► LocalCNN ──► 128-dim
     │      Gaussian mask @ cursor → weighted pool → FC
     │
     └──► DOM Heatmap ──► (H×W) float32
           scrape interactive elements
           Levenshtein score vs. query

Joint vector = [global(256) | local(128) | dom(W×H)]
     │
     ├──► grid_head    → WHERE to click  (10×10 = 100 cells)
     ├──► action_head  → WHAT action     (7 choices)
     └──► key_head     → WHICH key       (72 chars)
```

---

## Algorithm — Deep Q-Learning

**Q-network:** linear approximation

```
Q(s, a) = s @ W + b
```

**ε-greedy selection** — explore early, exploit later:

```
a = random          with prob ε
a = argmax Q(s, ·)  otherwise
```

**Bellman target:**

```
y = r + γ · max_a' Q_target(s', a') · (1 - done)
```

**Loss and update:**

```
L      = ( Q(s,a) - y )²
W[:,a] -= lr · (Q(s,a) - y) · s
```

**Replay buffer** — stores `(s, a, r, s′, done)`, sampled randomly each step
→ breaks temporal correlation.

**Target network** — frozen copy, synced every C episodes
→ prevents the moving-target problem.

---

## MDP

| Symbol | Meaning |
|--------|---------|
| S | screenshot + DOM heatmap + cursor |
| A | {left_click, scroll_up, scroll_down, move, type_text, …} |
| R | shaped reward (URL change, DOM match, step penalty, …) |
| γ | 0.99 |

---

## Methods Comparison

| Method       | Updates        | Variance | Bias |
|--------------|----------------|----------|------|
| Monte Carlo  | End of episode | High     | None |
| TD(0)        | Every step     | Low      | High |
| Q-Learning   | Every step     | Low      | High |
| DQL (this)   | Every step     | Low      | High |

---

## Project Structure

```
webagent/
├── gui.py                       ← Streamlit control panel
├── train.py                     ← DQL training + task runner
├── dataset/
│   └── tasks.json               ← Training task list
├── checkpoints/                 ← Saved Q-network weights (.npz)
└── env/
    ├── screen_processor.py      PNG → (W,H,3) float32
    ├── global_cnn.py            5 conv layers → 256-dim
    ├── local_cnn.py             Gaussian attention → 128-dim
    ├── dom_feature_extractor.py DOM scrape + Levenshtein scoring
    ├── environment_handler.py   Perception–action loop
    ├── reward.py                Shaped R(s,a)
    └── miniwob_env.py           MiniWoB++ wrapper
```

---

## Install

```bash
pip install streamlit playwright pillow numpy
playwright install chromium
```

---

## Run

**GUI:**
```bash
streamlit run gui.py
```

**CLI — train then auto-run first task:**
```bash
python train.py --tasks dataset/tasks.json --episodes 300
```

**CLI — MiniWoB:**
```bash
# Serve MiniWoB locally:
git clone https://github.com/Farama-Foundation/miniwob-plusplus
cd miniwob-plusplus && python -m http.server 7878

# Train:
python train.py --miniwob --miniwob_tasks click-button click-link --episodes 200
```

**Run only — use a saved checkpoint:**
```bash
python train.py --run-only --resume checkpoints/final.npz \
                --run-url https://github.com --run-query "Click Sign in"
```

**Show browser during task execution:**
```bash
python train.py --tasks dataset/tasks.json --episodes 300 --show-browser
```

---

## Reward Table

| Signal              | Value  |
|---------------------|--------|
| URL changed         | +1.00  |
| Title matches query | +0.60  |
| DOM element match   | +0.40  |
| Input field focused | +0.20  |
| Scroll progress     | +0.10  |
| Step penalty        | −0.05  |
| Repeat action       | −0.10  |
| Timeout             | −1.00  |

*All values clipped to [−1, +1].*
