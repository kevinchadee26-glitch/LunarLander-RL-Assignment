import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from utils.config import Config

# --- 1. The Neural Network (The Brain) ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Input: 8 (State) -> Hidden: 64 -> Hidden: 64 -> Output: 4 (Actions)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        # ReLU activation allows the network to learn complex non-linear patterns
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. The DQN Agent (The Manager) ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(Config.DEVICE)
        
        # Hyperparameters from Config
        self.gamma = Config.GAMMA
        self.batch_size = Config.BATCH_SIZE
        self.lr = Config.LEARNING_RATE
        
        # Initialize Networks (Main and Target)
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict()) # Clone weights initially
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Initialize Replay Buffer (Memory)
        self.memory = deque(maxlen=100000)
        
        # Exploration Parameters (Epsilon-Greedy)
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY

    def get_action(self, state):
        # 1. Exploration: Pick a random action
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # 2. Exploitation: Ask the brain for the best action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def learn(self):
        # Don't train if we don't have enough data yet
        if len(self.memory) < self.batch_size:
            return

        # 1. Sample a random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors and move to Device (GPU/CPU)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 2. Compute Target Q-Values (Bellman Equation)
        # We use the Target Network for stability
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            q_targets = rewards + (self.gamma * next_q_values * (1 - dones))

        # 3. Compute Current Q-Values (Prediction)
        current_q_values = self.q_network(states).gather(1, actions)
        
        # 4. Calculate Loss and Optimize
        loss = nn.MSELoss()(current_q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 5. Decay Epsilon (Stop exploring gradually)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        # Sync the Target Network with the Main Network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def add_memory(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(self.q_network.state_dict())