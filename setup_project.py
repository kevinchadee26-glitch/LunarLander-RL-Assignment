import os

def create_project_structure():
    # 1. Define the folders we need
    folders = [
        "agents",
        "utils",
        "checkpoints",
        "results/figures",
        "results/videos"
    ]
    
    # 2. Define the empty files we need
    files = [
        "agents/__init__.py",
        "agents/dqn_agent.py",
        "agents/a2c_agent.py",
        "utils/__init__.py",
        "utils/config.py",
        "utils/plotting.py",
        "main.py",
        "README.md"
    ]
    
    print("Starting Project Setup...")
    
    # Create Folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created: {folder}")
        
    # Create Files
    for file in files:
        if not os.path.exists(file):
            with open(file, "w") as f:
                pass # Create empty file
            print(f"Created: {file}")
        else:
            print(f"Exists: {file}")
            
    print("\nProject Structure Ready! You can delete this script now.")

if __name__ == "__main__":
    create_project_structure()