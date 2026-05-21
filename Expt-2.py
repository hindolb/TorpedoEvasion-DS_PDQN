import numpy as np
import torch
import os

# Import the provided classes
from Environment_PDQN_DS import AuvEvasionEnv
from PDQN_DS import PDQNAgent

def evaluate_model():
    # Experimental setup defined in the paper
    episodes_per_scenario = 1000
    torpedo_counts = [1, 2, 3, 4, 5]
    distance_ranges = [
        (1500.0, 1800.0),
        (1800.0, 2100.0),
        (2100.0, 2400.0)
    ]
    
    checkpoint_path = "pdqn_ds_final.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found.")
        print("Please run Training_DS.py to completion before evaluating.")
        return

    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    
    print(f"{'='*80}")
    print(f"Starting Evaluation: {episodes_per_scenario} episodes per configuration")
    print(f"{'='*80}")

    for dist_range in distance_ranges:
        print(f"\n>>> Initial Distance Range: {dist_range[0]} - {dist_range[1]} m")
        print(f"{'-'*80}")
        print(f"{'Torps':<7} | {'Survival (%)':<14} | {'Mean Dist (m)':<20} | {'Pursuit Time (s)':<20}")
        print(f"{'-'*80}")
        
        for n_torps in torpedo_counts:
            # 1. Initialize the Environment
            env = AuvEvasionEnv(
                n_torpedoes=n_torps, 
                min_spawn_dist=dist_range[0], 
                max_spawn_dist=dist_range[1],
                mode="testing"
            )
            
            # Infer dimensions
            feature_dim = env.observation_space.shape[1]
            discrete_dim = env.action_space[0].nvec[0]
            continuous_dim = env.action_space[1].shape[0]

            # 2. Initialize the Agent
            agent = PDQNAgent(
                state_dim=feature_dim, 
                num_discrete=discrete_dim, 
                param_dim=continuous_dim,
                device='cpu' # Force CPU for deterministic evaluation
            )

            # 3. Load Checkpoint Weights
            agent.actor_encoder.load_state_dict(checkpoint["actor_encoder"])
            agent.critic_encoder.load_state_dict(checkpoint["critic_encoder"])
            agent.param_net.load_state_dict(checkpoint["param_net"])
            agent.q_net.load_state_dict(checkpoint["q_net"])

            # Set networks to evaluation mode
            agent.actor_encoder.eval()
            agent.critic_encoder.eval()
            agent.param_net.eval()
            agent.q_net.eval()

            # 4. Track Metrics
            survivals = 0

            # 5. Run Episodes
            for ep in range(episodes_per_scenario):
                state, _ = env.reset()
                done = False
                
                while not done:
                    # Pure exploitation: epsilon=0.0, noise_std=0.0
                    action = agent.select_action(state, epsilon=0.0, noise_std=0.0)
                    #decoy1, decoy2 = action[0]
                    #if decoy1 > 0 or decoy2 > 0:
                    #    print("Decoy Launched: ", decoy1, decoy2)
                    state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                # Record metrics based on outcome
                if info.get("survived", False):
                    survivals += 1

            # 6. Aggregate Statistics
            survival_rate = (survivals / episodes_per_scenario) * 100.0
                    
            print(f"{n_torps:<7} | {survival_rate:>12.2f} % ")

if __name__ == "__main__":
    evaluate_model()
