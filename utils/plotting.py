import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curve(scores, filename="results/figures/learning_curve.png"):
    """
    Plots the learning curve (Reward vs Episodes).
    - Light Blue: Raw score per episode (high variance).
    - Dark Blue: Moving average over 20 episodes (trend).
    """
    
    # 1. Create the Figure
    plt.figure(figsize=(10, 5))
    
    # 2. Plot Raw Scores (Light Cyan)
    # Alpha=0.3 makes it transparent so it doesn't clutter the view
    plt.plot(scores, color='cyan', alpha=0.3, label='Raw Score')
    
    # 3. Calculate Moving Average (Smoothing)
    window_size = 20
    avg_scores = []
    
    # This loop calculates the average of the last 'window_size' scores
    for i in range(len(scores)):
        if i < window_size:
            # If we don't have enough data yet, average whatever we have
            avg_scores.append(np.mean(scores[:i+1]))
        else:
            # Otherwise, average the last 20 scores
            avg_scores.append(np.mean(scores[i-window_size+1:i+1]))
            
    # 4. Plot Moving Average (Solid Blue)
    plt.plot(avg_scores, color='blue', label='Moving Average (20 eps)')
    
    # 5. Labels and Titles
    plt.title('Agent Learning Curve (LunarLander-v3)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()             # Shows the label box
    plt.grid(True, alpha=0.3) # Adds a faint grid for readability
    
    # 6. Save the File
    # Check if the folder exists, if not, create it
    if not os.path.exists("results/figures"):
        os.makedirs("results/figures")
        
    plt.savefig(filename)
    plt.close() # Close the plot to free up memory
    print(f"Graph saved to {filename}")