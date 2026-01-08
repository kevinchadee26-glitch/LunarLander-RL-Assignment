# Lunar Lander Reinforcement Learning (DQN vs A2C)

This project implements and compares two Reinforcement Learning algorithms—**Deep Q-Network (DQN)** and **Actor-Critic (A2C)**—to solve the `LunarLander-v3` environment from Gymnasium.

The goal is to train an agent to safely land a spacecraft between two flags without crashing, using PyTorch.

## Project Structure

```text
├── agents/
│   ├── dqn_agent.py       # Deep Q-Network implementation (Replay Buffer + Target Net)
│   ├── a2c_agent.py       # Actor-Critic implementation (Shared Network)
│   └── __init__.py
├── utils/
│   ├── config.py          # Hyperparameters (LR, Gamma, Epsilon, etc.)
│   ├── plotting.py        # Tools for plotting learning curves
│   └── __init__.py
├── checkpoints/           # Saved model weights (.pth files)
├── results/               # Generated graphs and video recordings
├── main.py                # Main training loop and evaluation script
└── requirements.txt       # Dependencies
```
