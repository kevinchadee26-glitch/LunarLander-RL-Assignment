import gymnasium as gym
import numpy as np
import os
from gymnasium.wrappers import RecordVideo
from utils.config import Config
from agents.dqn_agent import DQNAgent
from agents.a2c_agent import A2CAgent
from utils.plotting import plot_learning_curve

def train():
    print(f"Starting Training on Device: {Config.DEVICE}")
    
    # 1. Initialize Environment
    # We use "human" to see it live, or "rgb_array" to run faster in background
    env = gym.make(Config.ENV_NAME, render_mode=Config.RENDER_MODE)
    
    # --- 2. SELECT YOUR AGENT HERE ---
    # Uncomment the one you want to train:
    
    agent = DQNAgent(state_size=8, action_size=4)
    # agent = A2CAgent(state_size=8, action_size=4)
    
    # Automatically detect name for file saving
    agent_name = "DQN" if isinstance(agent, DQNAgent) else "A2C"
    print(f"Agent Selected: {agent_name}")
    
    scores = []
    
    # 3. Training Loop
    for episode in range(Config.NUM_EPISODES):
        state, info = env.reset(seed=Config.SEED + episode)
        done = False
        total_reward = 0
        
        while not done:
            # A. Get Action
            action = agent.get_action(state)
            
            # B. Take Step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # C. Save Memory & Learn
            agent.add_memory(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            
        # D. Update Target Network (DQN Only)
        # We do this every 10 episodes to keep training stable
        if episode % 10 == 0:
            agent.update_target_network()
            
        scores.append(total_reward)
        
        # Calculate trailing average for the logs
        avg_score = np.mean(scores[-100:])
        
        print(f"Episode {episode+1} | Score: {total_reward:.2f} | Avg: {avg_score:.2f} | Eps: {agent.epsilon:.2f}")

    env.close()
    
    # 4. Save Results
    # Save the Graph
    plot_learning_curve(scores, filename=f"results/figures/{agent_name}_curve.png")
    
    # Save the Brain (Model Weights)
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    
    save_path = f"checkpoints/{agent_name}_final.pth"
    agent.save(save_path)
    print(f"Model saved to {save_path}")
    
    return agent, agent_name

def record_video(agent, agent_name):
    """
    Records one episode of the trained agent flying.
    """
    print("🎥 Recording final video...")
    
    # We need "rgb_array" mode to capture the video frames
    env = gym.make(Config.ENV_NAME, render_mode="rgb_array")
    
    # Wrap the env to auto-save video
    video_path = "results/videos"
    env = RecordVideo(
        env, 
        video_folder=video_path,
        name_prefix=f"{agent_name}_video",
        disable_logger=True
    )
    
    state, _ = env.reset(seed=Config.SEED + 999) # Different seed for testing
    done = False
    
    # Force Exploitation (Turn off randomness to show best behavior)
    agent.epsilon = 0.0
    
    while not done:
        action = agent.get_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    print(f"Video saved to {video_path}/{agent_name}_video-episode-0.mp4")

if __name__ == "__main__":
    # 1. Train the agent
    trained_agent, name = train()
    
    # 2. Record a victory lap video
    # (Only runs if you have moviepy installed)
    try:
        record_video(trained_agent, name)
    except Exception as e:
        print(f"Could not record video (missing moviepy?): {e}")