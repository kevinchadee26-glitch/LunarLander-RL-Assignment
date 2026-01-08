import torch

class Config:
    # --- 1. Environment Settings ---
    ENV_NAME = "LunarLander-v3"
    RENDER_MODE = "human"  
    
    # --- 2. Device Configuration ---
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
        
    SEED = 42
    
    # --- 3. Training Loop Parameters ---
    NUM_EPISODES = 100      # Total number of episodes to run
    MAX_STEPS = 1000        # Force quit episode if it takes too long
    
    # --- 4. Model Hyperparameters ---
    LEARNING_RATE = 1e-3    # (0.001) Controls how drastically the brain updates
    GAMMA = 0.99            # Discount Factor: How much we care about future rewards
    BATCH_SIZE = 64         # How many memories to replay at once
    
    # --- 5. Exploration Settings (DQN Only) ---
    EPSILON_START = 1.0     # Start 100% random (Exploration)
    EPSILON_DECAY = 0.995   # Reduce randomness by 0.5% every update
    EPSILON_MIN = 0.01      # Stop decaying when we reach 1% randomness