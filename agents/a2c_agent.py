import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.config import Config

# --- 1. The Actor-Critic Network ---
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticNetwork, self).__init__()
        # Common Feature Extractor (Shared layers)
        self.fc1 = nn.Linear(state_size, 128)
        
        # Actor Head: Decides WHICH action to take (Probability distribution)
        self.actor = nn.Linear(128, action_size)
        
        # Critic Head: Decides HOW GOOD the state is (Value estimation)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        
        # Actor: Softmax ensures probabilities sum to 1
        action_probs = torch.softmax(self.actor(x), dim=-1)
        
        # Critic: Linear output (Estimate of Value)
        state_value = self.critic(x)
        
        return action_probs, state_value

# --- 2. The A2C Agent ---
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.device = torch.device(Config.DEVICE)
        self.gamma = Config.GAMMA
        self.lr = Config.LEARNING_RATE
        
        # Initialize the shared network
        self.network = ActorCriticNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        # Rollout Buffer (Stores data for the current batch only)
        self.memory = [] 
        
        # Placeholder for compatibility with main.py (A2C doesn't use Epsilon)
        self.epsilon = 0.0

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Ask the Actor for probabilities
        action_probs, _ = self.network(state_tensor)
        
        # Sample an action based on those probabilities (Stochastic Policy)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        return action.item()

    def learn(self):
        # A2C updates more frequently (e.g., every 5 steps or end of episode)
        if len(self.memory) < 5:
            return

        states, actions, rewards, next_states, dones = zip(*self.memory)
        
        # Convert to Tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 1. Get predictions from the network
        action_probs, values = self.network(states)
        _, next_values = self.network(next_states)
        
        # 2. Calculate Targets (Bootstrapping)
        # Target = Reward + Gamma * Next_Value (if not done)
        target_values = rewards + (self.gamma * next_values * (1 - dones))
        
        # 3. Calculate Advantage
        # Advantage = (Actual Return) - (Critic's Prediction)
        # It tells us: "Was this action better than we expected?"
        advantage = target_values - values
        
        # 4. Actor Loss (Policy Gradient)
        # We encourage actions that had a high positive Advantage
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions.squeeze())
        actor_loss = -(log_probs * advantage.detach()).mean()
        
        # 5. Critic Loss (MSE)
        # The Critic tries to make its value prediction closer to the actual return
        critic_loss = nn.MSELoss()(values, target_values.detach())
        
        # 6. Total Loss & Update
        total_loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Clear memory (A2C is on-policy, so we discard old data immediately)
        self.memory = []

    # --- Compatibility Methods (So main.py works for both agents) ---
    def add_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_network(self):
        pass # A2C doesn't use a target network, so this does nothing

    def save(self, filename):
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        self.network.load_state_dict(torch.load(filename))