# 🧠 RL Dungeon Agent — Streamlit App

A Reinforcement Learning (Q-Learning) agent that learns to navigate a dungeon, with a full Streamlit UI.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```
rl_dungeon/
├── app.py            # Main Streamlit app (all-in-one)
├── requirements.txt  # Python dependencies
└── README.md
```

## 🖥️ Recommended IDE
**VS Code** — Free, lightweight, great Python + Streamlit support.

Install the **Python** and **Pylance** extensions in VS Code.

## 📊 Features
- 3 Dungeon levels: Beginner (4×4), Intermediate (6×6), Advanced (8×8)
- Manual play + trained agent auto-play
- Live RL metric graphs:
  - Cumulative Reward per Episode
  - Steps per Episode
  - ε (Epsilon) Decay
  - TD Error convergence
  - Policy Convergence
  - Win Rate (rolling window)
  - Q-Value Heatmap
  - Learned Policy Arrows
- Adjustable hyperparameters (α, γ, ε decay, episodes)

## 🧮 Algorithm
Q-Learning with ε-greedy exploration.
`Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') − Q(s,a)]`
