"""
train.py
Deep Q-Learning (DQL) for the Web Agent — pure Gymnasium, no Playwright.

Pipeline
--------
  1. Train Q-network on MiniWoB++ tasks via Gymnasium API
  2. Save final checkpoint
  3. Load checkpoint → run a real task greedily
  4. Print step-by-step results + final outcome

Algorithm
---------
  Q(s,a) approximated by a linear network: Q = s @ W + b

  Each episode:
    - ε-greedy action selection (ε decays over training)
    - Store (s, a, r, s', done) in replay buffer
    - Sample mini-batch → compute Bellman target
    - Gradient step on MSE loss
    - Every C episodes: sync target network

  Bellman target:
    y = r                            if done
    y = r + γ · max_a' Q_tgt(s',a') otherwise

  Loss:  L = ( Q(s,a) − y )²

Usage
-----
  # Train on default tasks then run:
  python train.py --episodes 300

  # Specify tasks:
  python train.py --tasks click-button click-link focus-text --episodes 200

  # Run only with saved checkpoint:
  python train.py --run-only --resume checkpoints/final.npz --run-task click-button
"""

import argparse
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "env"))
from environment_handler import ACTIONS, action_to_gym
from miniwob_env         import MiniWoBGymEnv, EASY_TASKS


# ── Q-Network ─────────────────────────────────────────────────────────────────

class QNetwork:
    """
    Linear Q-network:  Q(s, a) = s @ W + b

    Parameters
    ----------
    state_dim  : int
    n_actions  : int
    lr         : float
    seed       : int
    """

    def __init__(self, state_dim, n_actions, lr=1e-3, seed=0):
        rng     = np.random.default_rng(seed)
        self.W  = rng.normal(0, np.sqrt(2.0 / state_dim),
                             (state_dim, n_actions)).astype(np.float32)
        self.b  = np.zeros(n_actions, dtype=np.float32)
        self.lr = lr

    def predict(self, state):
        """Q(s, ·) → (n_actions,) float32"""
        return state @ self.W + self.b

    def update(self, state, action_idx, target):
        """One gradient step on L = (Q(s,a) - target)²"""
        q_vals                = self.predict(state)
        error                 = q_vals[action_idx] - target
        self.W[:, action_idx] -= self.lr * error * state
        self.b[action_idx]    -= self.lr * error

    def copy_weights_from(self, other):
        self.W[:] = other.W
        self.b[:] = other.b


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self._buf, min(batch_size, len(self._buf)))
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a),
                np.array(r, dtype=np.float32),
                np.array(ns), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self._buf)


# ── DQL helpers ───────────────────────────────────────────────────────────────

def dql_update(q_net, q_target, replay, batch_size, gamma):
    if len(replay) < batch_size:
        return
    states, actions, rewards, next_states, dones = replay.sample(batch_size)
    for i in range(len(states)):
        q_next = q_target.predict(next_states[i])
        target = rewards[i] + gamma * q_next.max() * (1.0 - dones[i])
        q_net.update(states[i], int(actions[i]), float(target))


def epsilon_greedy(q_net, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return int(q_net.predict(state).argmax())


# ── Episode runner (pure Gymnasium) ──────────────────────────────────────────

def run_episode_dql(q_net, replay, epsilon, task_name, max_steps, seed):
    """
    One MiniWoB++ episode via Gymnasium.
    Uses .send(ai) to inject the Q-network / epsilon-greedy action into the env
    so the executed action matches the one stored in the replay buffer.

    Returns (total_reward, steps, success).
    """
    env      = MiniWoBGymEnv(task_name=task_name, max_steps=max_steps, seed=seed)
    total_r  = 0.0
    steps    = 0
    success  = False
    prev_joint = None

    try:
        gen = env.run()
        # First yield: perception only (no reward yet)
        step_data = next(gen)
        while True:
            joint = step_data["joint_vec"]
            ai    = epsilon_greedy(q_net, joint, epsilon, len(ACTIONS))
            try:
                # Send chosen action → env executes it → yields result with reward
                result = gen.send(ai)
            except StopIteration:
                break
            r    = result["miniwob_reward"]
            done = result["done"]

            if prev_joint is not None:
                replay.push(prev_joint, ai, r, joint, float(done))
            prev_joint = joint
            total_r   += r
            steps     += 1
            if r > 0:
                success = True
            if done:
                break
            try:
                step_data = next(gen)
            except StopIteration:
                break
    except Exception as e:
        print(f"  episode error ({task_name}): {e}")
    finally:
        env.close()

    return total_r, steps, success


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(q_net, path):
    np.savez(path, W=q_net.W, b=q_net.b)
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(q_net, path):
    d = np.load(path)
    q_net.W[:] = d["W"]
    q_net.b[:] = d["b"]
    print(f"  [ckpt] loaded ← {path}")


# ── Run task with trained weights ─────────────────────────────────────────────

def run_task(q_net, task_name, max_steps=10, render_mode=None, seed=0):
    """
    Execute one task greedily (ε=0) with the trained Q-network.
    Prints each step and returns the final result dict.
    """
    print(f"\n{'─'*62}")
    print(f"  TASK EXECUTION  (greedy policy, ε=0)")
    print(f"  Task  : {task_name}")
    print(f"{'─'*62}")

    env      = MiniWoBGymEnv(task_name=task_name, max_steps=max_steps,
                             render_mode=render_mode, seed=seed)
    total_r  = 0.0
    steps    = 0
    success  = False
    actions_taken = []

    try:
        gen = env.run()
        step_data = next(gen)
        while True:
            joint    = step_data["joint_vec"]
            q_vals   = q_net.predict(joint)
            ai       = int(q_vals.argmax())          # greedy
            act_name = ACTIONS[ai]
            try:
                result = gen.send(ai)
            except StopIteration:
                break
            r    = result["miniwob_reward"]
            done = result["done"]

            total_r      += r
            steps        += 1
            actions_taken.append(act_name)
            if r > 0:
                success = True

            top3  = q_vals.argsort()[-3:][::-1]
            q_str = "  ".join(f"{ACTIONS[i]}={q_vals[i]:+.3f}" for i in top3)

            print(f"  [{result['step']:>2}] {act_name:<14}")
            print(f"       Q-top3 : {q_str}")
            print(f"       reward={r:+.3f}  G={total_r:.3f}"
                  f"  {'✓ terminal' if done and success else ''}")

            if done:
                break
            try:
                step_data = next(gen)
            except StopIteration:
                break
    finally:
        env.close()

    result = {
        "success":       success,
        "total_reward":  total_r,
        "steps":         steps,
        "actions_taken": actions_taken,
    }

    print(f"\n{'─'*62}")
    print(f"  RESULT")
    print(f"  Success      : {'✓ YES' if success else '✗ NO (max steps or stuck)'}")
    print(f"  Total Return : {total_r:.4f}")
    print(f"  Steps        : {steps}")
    print(f"  Actions      : {' → '.join(actions_taken)}")
    print(f"{'─'*62}\n")

    return result


# ── Training loop ─────────────────────────────────────────────────────────────

def train(tasks=None, n_episodes=300, max_steps=10, lr=1e-3, gamma=0.99,
          epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
          batch_size=32, target_update=10, buffer_capacity=10_000,
          save_every=100, checkpoint_dir="checkpoints", resume=None, seed=42,
          run_after=True, run_task_name=None, render_mode=None):
    """
    DQL training loop — pure Gymnasium.

    After training, optionally runs one task with the learned weights.
    """
    if tasks is None:
        tasks = EASY_TASKS[:3]  # default: click-button, click-link, click-checkboxes

    os.makedirs(checkpoint_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    # Build one env just to get the joint_dim
    _probe = MiniWoBGymEnv(tasks[0], max_steps=1, seed=seed)
    # Run one step to find joint vector size
    joint_dim = None
    try:
        for sd in _probe.run():
            joint_dim = sd["joint_vec"].shape[0]
            break
    except Exception:
        pass
    finally:
        _probe.close()

    if joint_dim is None:
        print("ERROR: could not determine state dimension. "
              "Is 'miniwob' package installed?  pip install miniwob")
        sys.exit(1)

    print(f"State dim : {joint_dim}")

    q_net    = QNetwork(joint_dim, len(ACTIONS), lr=lr, seed=seed)
    q_target = QNetwork(joint_dim, len(ACTIONS), lr=lr, seed=seed + 1)
    q_target.copy_weights_from(q_net)
    replay   = ReplayBuffer(capacity=buffer_capacity)

    if resume:
        load_checkpoint(q_net, resume)
        q_target.copy_weights_from(q_net)

    epsilon = epsilon_start
    rng     = random.Random(seed)

    print(f"DQL  |  episodes={n_episodes}  lr={lr}  γ={gamma}")
    print(f"ε    :  {epsilon_start} → {epsilon_end}  (decay={epsilon_decay})")
    print(f"buf  :  capacity={buffer_capacity}  batch={batch_size}")
    print(f"tasks:  {tasks}")
    print()

    for ep in range(1, n_episodes + 1):
        t0        = time.time()
        task_name = rng.choice(tasks)

        total_r, steps, success = run_episode_dql(
            q_net, replay, epsilon, task_name, max_steps, seed=ep)

        dql_update(q_net, q_target, replay, batch_size, gamma)

        if ep % target_update == 0:
            q_target.copy_weights_from(q_net)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        print(f"Ep {ep:>4}  G={total_r:+.3f}  steps={steps:>3}"
              f"  ε={epsilon:.3f}  buf={len(replay):>5}"
              f"  {'✓' if success else '✗'}  {time.time()-t0:.1f}s"
              f"  [{task_name}]")

        if ep % save_every == 0:
            save_checkpoint(q_net, os.path.join(checkpoint_dir, f"ep_{ep:05d}.npz"))

    save_checkpoint(q_net, os.path.join(checkpoint_dir, "final.npz"))
    print("\nTraining complete.")

    if run_after:
        t = run_task_name or tasks[0]
        run_task(q_net, t, max_steps=max_steps, render_mode=render_mode, seed=seed)

    return q_net


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Web Agent — DQL Training + Task Runner  (Gymnasium only)")

    p.add_argument("--tasks", nargs="+",
                   default=["click-button", "click-link", "focus-text"],
                   help="MiniWoB++ task names to train on")

    p.add_argument("--episodes",      type=int,   default=300)
    p.add_argument("--max_steps",     type=int,   default=10)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--epsilon_start", type=float, default=1.0)
    p.add_argument("--epsilon_end",   type=float, default=0.05)
    p.add_argument("--epsilon_decay", type=float, default=0.995)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--target_update", type=int,   default=10)
    p.add_argument("--buffer",        type=int,   default=10_000)
    p.add_argument("--seed",          type=int,   default=42)

    p.add_argument("--save_every",     type=int,  default=100)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--resume",         default=None)

    p.add_argument("--no-run",   action="store_true",
                   help="Skip task execution after training")
    p.add_argument("--run-only", action="store_true",
                   help="Load checkpoint and run one task — no training")
    p.add_argument("--run-task", default=None,
                   help="Task name to run after training (default: first training task)")
    p.add_argument("--show-browser", action="store_true",
                   help="Render browser window during task execution")

    args = p.parse_args()
    render_mode = "human" if args.show_browser else None

    if args.run_only:
        if not args.resume:
            print("--run-only requires --resume <checkpoint.npz>")
            sys.exit(1)
        task = args.run_task or args.tasks[0]
        # Need joint_dim — probe the env
        _probe = MiniWoBGymEnv(task, max_steps=1, seed=args.seed)
        joint_dim = None
        try:
            for sd in _probe.run():
                joint_dim = sd["joint_vec"].shape[0]
                break
        finally:
            _probe.close()
        if joint_dim is None:
            print("ERROR: could not determine state dimension. "
                  "Is 'miniwob' package installed?  pip install miniwob")
            sys.exit(1)
        q_net = QNetwork(joint_dim, len(ACTIONS), lr=args.lr)
        load_checkpoint(q_net, args.resume)
        run_task(q_net, task, max_steps=args.max_steps,
                 render_mode=render_mode, seed=args.seed)
        sys.exit(0)

    train(
        tasks=args.tasks,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        target_update=args.target_update,
        buffer_capacity=args.buffer,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        seed=args.seed,
        run_after=not args.no_run,
        run_task_name=args.run_task,
        render_mode=render_mode,
    )
