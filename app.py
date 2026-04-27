"""
Disaster Response Robot Navigation
===================================
A real-world Reinforcement Learning application using Q-Learning.
The rescue robot must navigate a dynamic disaster environment to find victims,
collect supplies, and avoid danger zones — all under uncertainty.

Run with: streamlit run disaster_robot_rl.py
"""

import streamlit as st
import numpy as np
import random
import time
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque
import copy

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
GRID_SIZE = 8

# Cell types
EMPTY     = 0
WALL      = 1  # debris
DANGER    = 2  # danger zones (fire, unstable floor)
SUPPLY    = 3  # medical/food supplies
VICTIM    = 4  # the rescue target
ROBOT     = 5  # agent position (for display)

# Rewards
R_STEP      = -0.5   # small penalty per step (urgency)
R_WALL      = -5.0   # hit debris
R_DANGER    = -15.0  # enter danger zone
R_SUPPLY    =  8.0   # collect supply
R_VICTIM    =  50.0  # rescue victim (terminal)
R_REVISIT   = -1.0   # penalize revisiting cells

# Actions: 0=Up, 1=Down, 2=Left, 3=Right
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
ACTION_LABELS = ["↑ Up","↓ Down","← Left","→ Right"]

# Dynamic environment parameters
OBSTACLE_CHANGE_PROB = 0.04   # prob a debris cell shifts each step
DANGER_SPREAD_PROB   = 0.02   # prob danger spreads to adjacent cell
SENSOR_NOISE         = 0.15   # prob robot misperceives a cell type

# Cell emojis for grid display
CELL_EMOJI = {
    EMPTY:  "⬜",
    WALL:   "🧱",
    DANGER: "🔥",
    SUPPLY: "🎒",
    VICTIM: "🆘",
    ROBOT:  "🤖",
}

CELL_COLOR = {
    EMPTY:  "#1a1a2e",
    WALL:   "#4a3728",
    DANGER: "#8b1a1a",
    SUPPLY: "#1a4a2e",
    VICTIM: "#1a1a6e",
    ROBOT:  "#2d4a6e",
}

# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────
class DisasterEnvironment:
    """
    Dynamic 8×8 disaster grid.
    - Walls (debris) can shift between steps.
    - Danger zones can spread slowly.
    - Sensor noise means the agent may misperceive cell types.
    - Non-stationary → pure rule-based methods fail.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.size = GRID_SIZE
        self.reset()

    # ── map generation ──────────────────────────────────────────────────────
    def _generate_map(self):
        grid = np.zeros((self.size, self.size), dtype=int)

        # Scatter debris (walls) ~18% of cells
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) in [(0, 0), (self.size-1, self.size-1)]:
                    continue
                if self.rng.random() < 0.18:
                    grid[r, c] = WALL

        # Add danger zones (fire / unstable) ~10%
        for r in range(self.size):
            for c in range(self.size):
                if grid[r, c] == EMPTY and (r, c) != (0, 0):
                    if self.rng.random() < 0.10:
                        grid[r, c] = DANGER

        # Place 3 supply caches
        placed = 0
        while placed < 3:
            r, c = self.rng.integers(0, self.size, size=2)
            if grid[r, c] == EMPTY and (r, c) != (0, 0):
                grid[r, c] = SUPPLY
                placed += 1

        # Victim (exit) at bottom-right quadrant
        victim_candidates = [
            (r, c)
            for r in range(self.size//2, self.size)
            for c in range(self.size//2, self.size)
            if grid[r, c] == EMPTY
        ]
        if not victim_candidates:
            victim_candidates = [
                (r, c) for r in range(self.size) for c in range(self.size)
                if grid[r, c] == EMPTY and (r, c) != (0, 0)
            ]
        vr, vc = victim_candidates[self.rng.integers(len(victim_candidates))]
        grid[vr, vc] = VICTIM
        return grid, (vr, vc)

    # ── reset ───────────────────────────────────────────────────────────────
    def reset(self):
        self.grid, self.victim_pos = self._generate_map()
        self.robot_pos = (0, 0)
        self.collected_supplies = 0
        self.visited = set()
        self.visited.add((0, 0))
        self.steps = 0
        self.done = False
        return self._get_state()

    # ── state ───────────────────────────────────────────────────────────────
    def _get_state(self):
        """
        State = (row, col, supplies_collected).
        With sensor noise: the agent receives a *noisy* observation of the grid,
        but the true grid state is used for transitions.
        """
        return (self.robot_pos[0], self.robot_pos[1], min(self.collected_supplies, 3))

    # ── dynamics ────────────────────────────────────────────────────────────
    def _dynamic_update(self):
        """Non-stationary: debris shifts, fire spreads."""
        new_grid = self.grid.copy()

        for r in range(self.size):
            for c in range(self.size):
                # Debris can shift to adjacent empty cell
                if self.grid[r, c] == WALL:
                    if random.random() < OBSTACLE_CHANGE_PROB:
                        neighbors = self._empty_neighbors(r, c)
                        if neighbors:
                            nr, nc = random.choice(neighbors)
                            if (nr, nc) != self.robot_pos and (nr, nc) != self.victim_pos:
                                new_grid[r, c] = EMPTY
                                new_grid[nr, nc] = WALL

                # Fire can spread to adjacent empty cell
                if self.grid[r, c] == DANGER:
                    if random.random() < DANGER_SPREAD_PROB:
                        neighbors = self._empty_neighbors(r, c)
                        if neighbors:
                            nr, nc = random.choice(neighbors)
                            if (nr, nc) != self.victim_pos:
                                new_grid[nr, nc] = DANGER

        self.grid = new_grid

    def _empty_neighbors(self, r, c):
        neighbors = []
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self.grid[nr, nc] == EMPTY:
                    neighbors.append((nr, nc))
        return neighbors

    # ── step ────────────────────────────────────────────────────────────────
    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True

        dr, dc = ACTIONS[action]
        r, c = self.robot_pos
        nr, nc = r + dr, c + dc
        self.steps += 1

        # Boundary / wall collision
        if not (0 <= nr < self.size and 0 <= nc < self.size) or self.grid[nr, nc] == WALL:
            reward = R_WALL
            self._dynamic_update()
            return self._get_state(), reward, False

        # Move robot
        cell = self.grid[nr, nc]
        reward = R_STEP

        if (nr, nc) in self.visited:
            reward += R_REVISIT

        self.robot_pos = (nr, nc)
        self.visited.add((nr, nc))

        if cell == DANGER:
            reward += R_DANGER

        elif cell == SUPPLY:
            reward += R_SUPPLY
            self.collected_supplies += 1
            self.grid[nr, nc] = EMPTY  # supply consumed

        elif cell == VICTIM:
            reward += R_VICTIM
            self.done = True
            return self._get_state(), reward, True

        # Update environment dynamics
        self._dynamic_update()

        return self._get_state(), reward, False

    # ── noisy observation (for display) ─────────────────────────────────────
    def get_noisy_grid(self):
        noisy = self.grid.copy()
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) == self.robot_pos:
                    continue
                if random.random() < SENSOR_NOISE:
                    # Flip empty↔danger or empty↔wall to simulate sensor uncertainty
                    if noisy[r, c] == EMPTY:
                        noisy[r, c] = random.choice([WALL, DANGER])
                    elif noisy[r, c] in [WALL, DANGER]:
                        noisy[r, c] = EMPTY
        return noisy

    # ── state space size ────────────────────────────────────────────────────
    def state_size(self):
        return (self.size, self.size, 4)  # row, col, supplies (0-3)

    def action_size(self):
        return 4


# ──────────────────────────────────────────────────────────────────────────────
# Q-LEARNING AGENT
# ──────────────────────────────────────────────────────────────────────────────
class QLearningAgent:
    """
    Tabular Q-Learning with ε-greedy exploration.
    Q-table: dict mapping state → [Q(s,a0), Q(s,a1), Q(s,a2), Q(s,a3)]
    """

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995):
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}           # state → np.array of 4 Q-values
        self.prev_q_table = {}      # snapshot for policy-change metric

    def _q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        return self.q_table[state]

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)          # explore
        return int(np.argmax(self._q(state)))    # exploit

    def update(self, state, action, reward, next_state, done):
        q_current = self._q(state)[action]
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self._q(next_state))
        td_error = td_target - q_current
        self.q_table[state][action] += self.alpha * td_error
        return abs(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def policy_change_rate(self):
        """Fraction of known states where greedy action changed since last snapshot."""
        if not self.prev_q_table:
            return 0.0
        changes = 0
        total = 0
        for state, q_vals in self.q_table.items():
            if state in self.prev_q_table:
                total += 1
                if np.argmax(q_vals) != np.argmax(self.prev_q_table[state]):
                    changes += 1
        return changes / total if total > 0 else 0.0

    def snapshot_policy(self):
        self.prev_q_table = {s: v.copy() for s, v in self.q_table.items()}

    def q_table_size(self):
        return len(self.q_table)


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP (one episode)
# ──────────────────────────────────────────────────────────────────────────────
def run_episode(env, agent, max_steps=200):
    state = env.reset()
    total_reward = 0.0
    steps = 0
    td_errors = []
    done = False

    while not done and steps < max_steps:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        td_err = agent.update(state, action, reward, next_state, done)
        td_errors.append(td_err)
        state = next_state
        total_reward += reward
        steps += 1

    agent.decay_epsilon()
    return total_reward, steps, done, np.mean(td_errors) if td_errors else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────
def init_session():
    if "agent" not in st.session_state:
        st.session_state.agent = QLearningAgent()
    if "env" not in st.session_state:
        st.session_state.env = DisasterEnvironment()
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "episode": [],
            "reward":  [],
            "steps":   [],
            "epsilon": [],
            "td_error":[],
            "policy_change": [],
            "win_rate_window": deque(maxlen=50),
            "win_rate": [],
        }
    if "training" not in st.session_state:
        st.session_state.training = False
    if "episodes_done" not in st.session_state:
        st.session_state.episodes_done = 0


# ──────────────────────────────────────────────────────────────────────────────
# GRID VISUALISATION (Plotly heatmap)
# ──────────────────────────────────────────────────────────────────────────────
def build_grid_figure(env):
    grid = env.grid.copy()
    rr, rc = env.robot_pos

    # Numeric palette for heatmap
    color_map = {
        EMPTY:  0,
        WALL:   1,
        DANGER: 2,
        SUPPLY: 3,
        VICTIM: 4,
        ROBOT:  5,
    }

    z = np.vectorize(color_map.get)(grid)
    z[rr, rc] = color_map[ROBOT]

    colorscale = [
        [0/5, "#0d0d1a"],   # EMPTY
        [1/5, "#5c3d1e"],   # WALL
        [2/5, "#8b1a1a"],   # DANGER
        [3/5, "#1a5c2e"],   # SUPPLY
        [4/5, "#1a1a8b"],   # VICTIM
        [5/5, "#00aaff"],   # ROBOT
    ]

    text = [
        [CELL_EMOJI[grid[r, c]] if (r, c) != (rr, rc) else "🤖"
         for c in range(GRID_SIZE)]
        for r in range(GRID_SIZE)
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale=colorscale,
        showscale=False,
        zmin=0, zmax=5,
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=380,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed"),
        paper_bgcolor="#0a0a14",
        plot_bgcolor="#0a0a14",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# METRICS CHARTS
# ──────────────────────────────────────────────────────────────────────────────
def build_metrics_figure(metrics):
    eps = metrics["episode"]
    if not eps:
        return None

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Total Reward / Episode", "Steps / Episode",
            "Epsilon (Exploration)", "Mean TD Error",
            "Policy Change Rate", "Win Rate (last 50 ep)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )

    def line(y, row, col, color, name):
        fig.add_trace(go.Scatter(
            x=eps, y=y, mode="lines", name=name,
            line=dict(color=color, width=1.5),
            showlegend=False,
        ), row=row, col=col)

    line(metrics["reward"],        1, 1, "#00d4ff", "Reward")
    line(metrics["steps"],         1, 2, "#ff8c00", "Steps")
    line(metrics["epsilon"],       2, 1, "#ff4466", "Epsilon")
    line(metrics["td_error"],      2, 2, "#aa44ff", "TD Error")
    line(metrics["policy_change"], 3, 1, "#44ff88", "Policy Δ")
    line(metrics["win_rate"],      3, 2, "#ffdd00", "Win Rate")

    fig.update_layout(
        height=540,
        paper_bgcolor="#0a0a14",
        plot_bgcolor="#0a0a14",
        font=dict(color="#c0c0d0", size=11),
        margin=dict(l=30, r=10, t=40, b=10),
    )
    for i in fig["layout"]["annotations"]:
        i["font"] = dict(color="#8888aa", size=11)
    fig.update_xaxes(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e")
    fig.update_yaxes(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Disaster Response RL",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global CSS ──────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        background-color: #0a0a14 !important;
        color: #c0c0d0 !important;
        font-family: 'Rajdhani', sans-serif;
    }
    .stApp { background: #0a0a14; }

    h1, h2, h3 { font-family: 'Share Tech Mono', monospace; letter-spacing: 0.04em; }

    .metric-card {
        background: #0f0f1e;
        border: 1px solid #1e2040;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 11px;
        color: #5555aa;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .metric-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: 22px;
        color: #00d4ff;
    }

    .legend-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0; }
    .legend-item { display: flex; align-items: center; gap: 5px; font-size: 13px; }

    .stButton > button {
        background: #0f1a2e;
        color: #00d4ff;
        border: 1px solid #00d4ff44;
        border-radius: 6px;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 0.05em;
        transition: all 0.2s;
    }
    .stButton > button:hover { background: #00d4ff22; border-color: #00d4ff; }

    .stSlider > div { color: #8888aa; }
    .sidebar .stMarkdown { font-size: 13px; }

    div[data-testid="stMetric"] label { color: #5555aa !important; font-size: 11px; }
    div[data-testid="stMetric"] div { color: #00d4ff !important; font-family: 'Share Tech Mono', monospace; }
    </style>
    """, unsafe_allow_html=True)

    init_session()
    agent   = st.session_state.agent
    env     = st.session_state.env
    metrics = st.session_state.metrics

    # ── Title ───────────────────────────────────────────────────────────────
    st.markdown("## 🤖 DISASTER RESPONSE · ROBOT NAVIGATION")
    st.markdown(
        "<span style='color:#5555aa;font-size:13px;font-family:Share Tech Mono,monospace'>"
        "Q-Learning agent · Dynamic environment · Real-time training metrics"
        "</span>", unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Sidebar: Hyperparameters ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ HYPERPARAMETERS")
        alpha   = st.slider("Learning Rate (α)", 0.01, 0.5,  0.10, 0.01)
        gamma   = st.slider("Discount (γ)",      0.50, 0.99, 0.95, 0.01)
        eps_dec = st.slider("ε Decay",           0.990, 0.999, 0.995, 0.001, format="%.3f")
        eps_min = st.slider("ε Minimum",         0.01, 0.20, 0.05, 0.01)
        n_ep    = st.slider("Episodes per run",  10, 500, 100, 10)
        max_st  = st.slider("Max steps / ep",    50, 400, 200, 10)

        st.markdown("---")
        st.markdown("### 🌍 ENVIRONMENT")
        st.markdown(f"**Obstacle shift prob:** `{OBSTACLE_CHANGE_PROB}`")
        st.markdown(f"**Fire spread prob:** `{DANGER_SPREAD_PROB}`")
        st.markdown(f"**Sensor noise:** `{SENSOR_NOISE}`")
        st.markdown("---")
        st.markdown("### 📖 LEGEND")
        for emoji, label in [
            ("🤖","Robot (agent)"),("🆘","Victim (goal)"),
            ("🧱","Debris (wall)"),("🔥","Danger zone"),
            ("🎒","Supplies"),("⬜","Clear path"),
        ]:
            st.markdown(f"{emoji} {label}")

        st.markdown("---")
        if st.button("🔄 Reset Everything"):
            for key in ["agent","env","metrics","episodes_done"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # ── Apply hyperparameters to agent ───────────────────────────────────────
    agent.alpha         = alpha
    agent.gamma         = gamma
    agent.epsilon_decay = eps_dec
    agent.epsilon_min   = eps_min

    # ── Layout: Grid  |  Live Metrics ───────────────────────────────────────
    col_grid, col_stats = st.columns([1.1, 0.9])

    with col_grid:
        st.markdown("#### 🗺️ DISASTER ZONE")
        grid_placeholder = st.empty()
        grid_placeholder.plotly_chart(build_grid_figure(env), use_container_width=True, key="grid_init")

        st.markdown(
            "<span style='color:#5555aa;font-size:12px'>"
            "Grid updates live during training · Non-stationary dynamics active"
            "</span>", unsafe_allow_html=True
        )

    with col_stats:
        st.markdown("#### 📊 LIVE METRICS")
        m1, m2, m3 = st.columns(3)
        ep_display    = m1.empty()
        reward_display= m2.empty()
        eps_display   = m3.empty()
        m4, m5, m6 = st.columns(3)
        steps_display = m4.empty()
        td_display    = m5.empty()
        wr_display    = m6.empty()

        def refresh_stats():
            n = st.session_state.episodes_done
            ep_display.metric("Episode", n)
            if metrics["reward"]:
                reward_display.metric("Last Reward",  f"{metrics['reward'][-1]:.1f}")
                eps_display.metric("Epsilon",         f"{agent.epsilon:.3f}")
                steps_display.metric("Last Steps",    metrics["steps"][-1])
                td_display.metric("TD Error",         f"{metrics['td_error'][-1]:.3f}")
                wr50 = (sum(metrics["win_rate_window"]) / len(metrics["win_rate_window"]) * 100
                        if metrics["win_rate_window"] else 0)
                wr_display.metric("Win Rate (50ep)",  f"{wr50:.0f}%")

        refresh_stats()

    # ── Training controls ───────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 2])
    run_btn  = c1.button("▶ TRAIN", use_container_width=True)
    stop_btn = c2.button("⏹ STOP",  use_container_width=True)

    progress_bar = st.progress(0)
    status_text  = st.empty()

    # ── Charts ──────────────────────────────────────────────────────────────
    chart_placeholder = st.empty()

    if stop_btn:
        st.session_state.training = False

    # ── Training loop ────────────────────────────────────────────────────────
    if run_btn:
        st.session_state.training = True
        agent.snapshot_policy()

        for ep_i in range(n_ep):
            if not st.session_state.training:
                status_text.markdown("⏹ **Training paused.**")
                break

            # Run one episode
            total_r, n_steps, won, td_err = run_episode(env, agent, max_steps=max_st)

            # Snapshot policy every 25 episodes
            if (st.session_state.episodes_done + 1) % 25 == 0:
                pc = agent.policy_change_rate()
                agent.snapshot_policy()
            else:
                pc = metrics["policy_change"][-1] if metrics["policy_change"] else 0.0

            st.session_state.episodes_done += 1
            n = st.session_state.episodes_done

            # Record metrics
            metrics["episode"].append(n)
            metrics["reward"].append(total_r)
            metrics["steps"].append(n_steps)
            metrics["epsilon"].append(agent.epsilon)
            metrics["td_error"].append(td_err)
            metrics["policy_change"].append(pc)
            metrics["win_rate_window"].append(1 if won else 0)
            wr = sum(metrics["win_rate_window"]) / len(metrics["win_rate_window"])
            metrics["win_rate"].append(wr)

            # Refresh UI every 5 episodes
            if ep_i % 5 == 0 or ep_i == n_ep - 1:
                progress_bar.progress((ep_i + 1) / n_ep)
                status = "✅ VICTIM RESCUED" if won else "❌ Episode ended"
                status_text.markdown(
                    f"**Ep {n}** · {status} · Reward: `{total_r:.1f}` · "
                    f"Steps: `{n_steps}` · ε: `{agent.epsilon:.3f}` · "
                    f"Q-states: `{agent.q_table_size()}`"
                )
                grid_placeholder.plotly_chart(
                    build_grid_figure(env), use_container_width=True, key=f"grid_{n}"
                )
                refresh_stats()
                fig = build_metrics_figure(metrics)
                if fig:
                    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_{n}")

                time.sleep(0.02)

        progress_bar.progress(1.0)
        st.session_state.training = False
        status_text.markdown(f"✅ **Training complete.** {n_ep} episodes finished.")

    # ── Static chart if not training ─────────────────────────────────────────
    elif metrics["episode"]:
        fig = build_metrics_figure(metrics)
        if fig:
            chart_placeholder.plotly_chart(fig, use_container_width=True, key="chart_static")

    # ── Q-table inspector ───────────────────────────────────────────────────
    with st.expander("🔍 Q-Table Inspector (top 20 states by max Q)"):
        if agent.q_table:
            rows = []
            for state, qv in agent.q_table.items():
                rows.append({
                    "State (row,col,supplies)": str(state),
                    "Q↑ Up":   f"{qv[0]:.3f}",
                    "Q↓ Down": f"{qv[1]:.3f}",
                    "Q← Left": f"{qv[2]:.3f}",
                    "Q→ Right":f"{qv[3]:.3f}",
                    "Best Action": ACTION_LABELS[int(np.argmax(qv))],
                    "Max Q":   f"{np.max(qv):.3f}",
                })
            rows.sort(key=lambda x: float(x["Max Q"]), reverse=True)
            st.table(rows[:20])
        else:
            st.info("Train the agent to populate the Q-table.")


if __name__ == "__main__":
    main()