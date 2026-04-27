# 🚨 Disaster Response Robot Navigation (Reinforcement Learning)

An interactive **Reinforcement Learning (RL)** project where a robot learns to navigate a **dynamic disaster environment** using **Q-Learning**.

---

## 🧠 Overview

The agent (robot) operates in an **8×8 grid world** where it must:

* 🆘 Rescue a victim (goal)
* 🔥 Avoid danger zones (fire, unstable areas)
* 🧱 Navigate around obstacles (debris)
* 🎒 Collect supplies for bonus rewards
* ⏱️ Act efficiently (step penalty encourages speed)

The environment is **non-stationary**:

* Obstacles can move
* Danger zones can spread
* Sensor readings can be noisy

👉 This makes it a realistic RL problem, not rule-based.

---

## ⚙️ Algorithm

* **Q-Learning (Tabular Reinforcement Learning)**
* **ε-greedy policy** for exploration vs exploitation

The agent learns by updating Q-values:

```id="c0v72c"
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]
```

---

## 🎮 Environment

* Grid size: **8 × 8**
* State: `(row, col, supplies_collected)`
* Actions: Up, Down, Left, Right

### Reward System

| Action/Event      | Reward |
| ----------------- | ------ |
| Step taken        | -0.5   |
| Hit obstacle      | -5     |
| Enter danger zone | -15    |
| Collect supply    | +8     |
| Reach victim      | +50    |
| Revisit cell      | -1     |

---

## 📊 Features

* 📈 Real-time training dashboard
* 🧭 Dynamic environment (moving obstacles + spreading danger)
* 🎯 Reward-based learning behavior
* 📉 Metrics:

  * Total reward
  * Steps per episode
  * Epsilon (exploration)
  * TD error
  * Policy change rate
  * Win rate
* 🔍 Q-table inspection

---


## 🧠 What This Project Demonstrates

* Reinforcement Learning in **dynamic environments**
* Difference between **rule-based vs learning-based systems**
* Importance of **reward design**
* Handling **uncertainty and noise**
* Learning optimal policies through **trial and error**

---

## 🌍 Real-World Applications

* 🚑 Disaster response robots
* 🚗 Autonomous navigation
* 🚁 Drone path planning
* 📦 Warehouse robotics
* 🚚 Smart logistics systems

---

## ⚠️ Limitations

* Uses **tabular Q-learning** (not scalable)
* No deep learning (no neural networks)
* Performance varies due to randomness

---

## 🔮 Future Improvements

* Implement **Deep Q-Network (DQN)**
* Multi-agent coordination
* Better state representation
* Real-world simulation integration

---

## 👩‍💻 Author

**Riddhi Koli**
