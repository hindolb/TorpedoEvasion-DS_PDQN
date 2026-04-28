import numpy as np
import torch
from collections import deque
from Environment_PDQN_DS import AuvEvasionEnv
from PDQN_DS import PDQNAgent

# ==========================================
# 1. Hyperparameters & Curriculum Config
# ==========================================
# Exploration Settings
NOISE_START = 0.5
NOISE_END = 0.05
EPSILON_START = 1.0
EPSILON_END = 0.10
DECAY_EPISODES_PER_LEVEL = 20000 

# Dual Curriculum Stages: [Torpedoes, Min_Dist, Max_Dist, Target_Survival]
CURRICULUM_STAGES = [
    {"torps": 1, "dist": (2100, 2400), "target": 98.0}, # Level 1: 1 Torp (Long Range)
    {"torps": 1, "dist": (1500, 2400), "target": 98.0}, # Level 2: 1 Torp (Full Range)
    {"torps": 2, "dist": (1500, 2400), "target": 85.0}, # Level 3: 2 Torps (Full Range)
    {"torps": 3, "dist": (1500, 2400), "target": 75.0}  # Level 4: 3 Torps (Target)
]

def train_auv():
    stage_idx = 0
    current_stage = CURRICULUM_STAGES[stage_idx]
    
    # Initialize Env with Stage 0 settings
    env = AuvEvasionEnv(n_torpedoes=current_stage["torps"])
    # Custom distance setters (Add these attributes to your Env reset logic)
    env.min_spawn_dist = current_stage["dist"][0]
    env.max_spawn_dist = current_stage["dist"][1]
    
    feature_dim = env.observation_space.shape[1] 
    discrete_dim = env.action_space[0].nvec[0] 
    continuous_dim = env.action_space[1].shape[0]
    
    agent = PDQNAgent(feature_dim, discrete_dim, continuous_dim)
    
    survival_window = deque(maxlen=100)
    episodes_in_current_level = 0
    epsilon_start = EPSILON_START
    total_episodes = 0

    print(f"Starting Curriculum: Stage {stage_idx+1} | {current_stage['torps']} Torpedoes | Distance {current_stage['dist']}")

    training_active = True
    while training_active:
        # Reset Env with current curriculum distance range
        # (Ensure your Environment_PDQN_DS.py uses these variables in reset())
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # Calculate Linear Decay for the CURRENT curriculum level
        decay_fraction = min(1.0, episodes_in_current_level / DECAY_EPISODES_PER_LEVEL)
        current_noise = max(NOISE_END, NOISE_START - (NOISE_START - NOISE_END) * decay_fraction)
        current_epsilon = max(EPSILON_END, epsilon_start - (epsilon_start - EPSILON_END) * decay_fraction)
        
        while not done:
            action_tuple = agent.select_action(state, epsilon=current_epsilon, noise_std=current_noise)
            next_state, reward, terminated, truncated, info = env.step(action_tuple)
            done = terminated or truncated
            
            discrete_act, continuous_act = action_tuple
            agent.remember(state, discrete_act, continuous_act, reward, next_state, done)
            agent.learn()
            
            state = next_state
            episode_reward += reward
            step += 1
            
        # --- Metrics tracking ---
        won = info.get("won", False)
        survival_window.append(1 if won else 0)
        current_survival_rate = (sum(survival_window) / len(survival_window)) * 100.0 if len(survival_window) > 0 else 0.0

        if total_episodes % 20 == 0:    
            print(f"Ep: {total_episodes+1:04d} | Lvl: {stage_idx+1} | Reward: {episode_reward:>7.1f} | "
                  f"Survival: {current_survival_rate:>5.1f}% | Torps: {current_stage['torps']}")

        episodes_in_current_level += 1
        total_episodes += 1

        # ==========================================
        # 2. Dual Curriculum Advancement Logic
        # ==========================================
        if len(survival_window) == 100 and current_survival_rate >= current_stage["target"]:
            # Check if there are more stages left
            if stage_idx < len(CURRICULUM_STAGES) - 1:
                stage_idx += 1
                current_stage = CURRICULUM_STAGES[stage_idx]
                
                print(f"\n{'='*60}")
                print(f"  >>> ADVANCING TO STAGE {stage_idx+1}: {current_stage['torps']} Torp, Dist {current_stage['dist']} <<<")
                print(f"{'='*60}\n")
                
                # Re-instantiate/Update Environment
                env = AuvEvasionEnv(n_torpedoes=current_stage["torps"])
                env.min_spawn_dist = current_stage["dist"][0]
                env.max_spawn_dist = current_stage["dist"][1]

                # Clear Buffer to prevent old data from polluting new difficulty
                agent.memory.buffer.clear()

                # Warm-start exploration: Give the agent a 25% exploration boost to adapt
                epsilon_start = min(1.0, current_epsilon + 0.25)
                
                # Reset metrics for new level
                survival_window.clear() 
                episodes_in_current_level = 0 
            else:
                # Reached the end of the final stage
                print("All Curriculum Stages Completed!")
                training_active = False

    # --- 3. Save Final Models ---
    torch.save(agent.param_net.state_dict(), "pdqn_param_final.pth")
    torch.save(agent.q_net.state_dict(), "pdqn_q_final.pth")
    print(f"Training Complete. Models saved. Final Torpedo Count: {current_stage['torps']}")

if __name__ == "__main__":
    train_auv()