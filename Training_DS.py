import numpy as np
import torch
from collections import deque
from Environment_PDQN_DS import AuvEvasionEnv
from PDQN_DS import PDQNAgent

#Hyperparameters
NOISE_START = 0.5
NOISE_END = 0.05
EPSILON_START = 1.0
EPSILON_END = 0.10               # Lower from 0.1 to 0.05

# Exploration will be reset/decayed per curriculum level
DECAY_EPISODES_PER_LEVEL = 4000  # Increase from 800 to 1000

# ==========================================
# NEW: Curriculum Hyperparameters
# ==========================================
TARGET_SURVIVAL_RATE = 85.0      # Increase from 80.0 to 85.0
MAX_TORPEDOES = 3            # Final curriculum level

def train_auv():
    # 1. Start Curriculum at 1 Torpedo
    current_torpedoes = 1
    env = AuvEvasionEnv(n_torpedoes=current_torpedoes)
    
    # 2. Initialize Agent (DeepSets uses feature_dim, not the flat state_dim)
    # env.observation_space.shape is now (n_torpedoes, 20), so shape[1] is 20
    feature_dim = env.observation_space.shape[1] 
    discrete_dim = env.action_space[0].n
    continuous_dim = env.action_space[1].shape[0]
    
    agent = PDQNAgent(feature_dim, discrete_dim, continuous_dim)
    
    # Increased total episodes to give enough time to beat all 3 levels
    episodes = 3000 
    
    survival_window = deque(maxlen=100)
    best_survival_rate = 0.0
    episodes_in_current_level = 0
    epsilon_start = EPSILON_START

    print(f"Starting Curriculum Learning: Level 1 ({current_torpedoes} Torpedo)")
    
    #for ep in range(episodes):
    ep = 0
    while True:
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # Calculate decay based on time spent in the CURRENT curriculum level
        decay_fraction = min(1.0, episodes_in_current_level / DECAY_EPISODES_PER_LEVEL)
        current_noise = max(NOISE_END, NOISE_START - (NOISE_START - NOISE_END) * decay_fraction)
        current_epsilon = max(EPSILON_END, epsilon_start - (epsilon_start - EPSILON_END) * decay_fraction)
        #current_epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * decay_fraction)
        
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
            
        # Outcome Tracking
        hit = info.get("hit", False)
        won = info.get("won", False)
        
        if hit:
            outcome = "DESTROYED"
            survival_window.append(0)
        elif won:
            outcome = "EVADED"
            survival_window.append(1)
        else:
            outcome = "TIMEOUT"
            survival_window.append(0)
            
        current_survival_rate = (sum(survival_window) / len(survival_window)) * 100.0 if len(survival_window) > 0 else 0.0

        if ep % 10 == 0:    
            print(f"Ep: {ep+1:04d} | Level: {current_torpedoes} Torps | Steps: {step:03d} | Reward: {episode_reward:>8.2f} | "
                  f"Noise: {current_noise:.2f} | Eps: {current_epsilon:.2f} | "
                  f"Outcome: {outcome:<10} | Survival: {current_survival_rate:>5.1f}%")

        episodes_in_current_level += 1

        # ==========================================
        # NEW: Curriculum Advancement Logic
        # ==========================================
        # If the window is full and the agent meets the required survival rate...
        if len(survival_window) == 100 and current_survival_rate >= TARGET_SURVIVAL_RATE:
            
            # ...and we haven't reached the final level yet
            if current_torpedoes < MAX_TORPEDOES:
                current_torpedoes += 1
                print(f"\n{'='*50}")
                print(f"  >>> CURRICULUM ADVANCED! Moving to {current_torpedoes} Torpedoes <<<")
                print(f"{'='*50}\n")
                
                # Re-instantiate the environment with the new number of torpedoes
                env = AuvEvasionEnv(n_torpedoes=current_torpedoes)

                agent.memory.buffer.clear()

                # Do NOT reset to 1.0. "Warm-start" the exploration.
                # The agent relies on its learned policy 75% of the time, 
                # and explores 25% of the time to adapt to the new torpedo.
                epsilon_start -= 0.25     
                
                # Reset metrics so it doesn't instantly skip the next level
                survival_window.clear() 
                episodes_in_current_level = 0 
                best_survival_rate = 0.0 
                
            # If we are at the max level (3 Torpedoes), save the best models normally
            else:
                if current_survival_rate >= best_survival_rate:
                    best_survival_rate = current_survival_rate
                    torch.save(agent.param_net.state_dict(), f"pdqn_param_best_{current_torpedoes}t.pth")
                    torch.save(agent.q_net.state_dict(), f"pdqn_q_best_{current_torpedoes}t.pth")

        if ep%100 == 0:
                torch.save(agent.param_net.state_dict(), "pdqn_param_final.pth")
                torch.save(agent.q_net.state_dict(), "pdqn_q_final.pth")
        ep += 1
    print(f"Models saved successfully. Best {current_torpedoes}-Torpedo Survival Rate: {best_survival_rate:.1f}%")
    print("\nTraining Complete.")
    torch.save(agent.param_net.state_dict(), "pdqn_param_final.pth")
    torch.save(agent.q_net.state_dict(), "pdqn_q_final.pth")
    print(f"Models saved successfully. Best {current_torpedoes}-Torpedo Survival Rate: {best_survival_rate:.1f}%")

if __name__ == "__main__":
    train_auv()