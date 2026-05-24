import numpy as np
import torch
import os

from Environment_PDQN_DS_Old_Reward import AuvEvasionEnv
from PDQN_DS import PDQNAgent


def evaluate_model():
    # Experimental setup defined in the paper
    episodes_per_scenario = 1000
    torpedo_counts  = [1, 2, 3, 4, 5]
    distance_ranges = [
        (1500.0, 1800.0),
        (1800.0, 2100.0),
        (2100.0, 2400.0),
    ]

    # Evaluate all available checkpoints so results can be compared directly.
    # pdqn_ds_best_finetune.pth is written by the fine-tuning phase and is
    # expected to produce the highest in-distribution survival rates.
    checkpoint_paths = [
        "pdqn_ds_best_finetune.pth",   # fine-tuned — best for eval
        "pdqn_ds_final.pth",           # end-of-training weights
        "pdqn_ds_best_stage3.pth",     # best weights seen during stage 3
    ]

    for checkpoint_path in checkpoint_paths:
        if not os.path.exists(checkpoint_path):
            print(f"Skipping '{checkpoint_path}' — file not found.")
            continue

        print(f"\n{'#' * 80}")
        print(f"  Checkpoint : {checkpoint_path}")
        print(f"{'#' * 80}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device('cpu'),
            weights_only=False,
        )

        # Dimensions are always read from the checkpoint so they stay consistent
        # with the saved weights regardless of which eval configuration is running.
        feature_dim    = checkpoint.get("state_dim",    20)
        discrete_dim   = checkpoint.get("num_discrete", 37)
        continuous_dim = checkpoint.get("param_dim",     2)

        # One agent instance per checkpoint, reused across all (n_torps, dist)
        # configurations. The DeepSetEncoder mean-pools over the torpedo
        # dimension, so a single model handles n=1..5 without any padding.
        agent = PDQNAgent(
            state_dim    = feature_dim,
            num_discrete = discrete_dim,
            param_dim    = continuous_dim,
            device       = 'cpu',
        )

        agent.actor_encoder.load_state_dict(checkpoint["actor_encoder"])
        agent.critic_encoder.load_state_dict(checkpoint["critic_encoder"])
        agent.param_net.load_state_dict(checkpoint["param_net"])
        agent.q_net.load_state_dict(checkpoint["q_net"])

        agent.actor_encoder.eval()
        agent.critic_encoder.eval()
        agent.param_net.eval()
        agent.q_net.eval()

        print(f"  state_dim={feature_dim}, num_discrete={discrete_dim}, "
              f"param_dim={continuous_dim}")
        print(f"  {episodes_per_scenario} episodes per configuration\n")

        for dist_range in distance_ranges:
            print(f">>> Initial Distance Range: {dist_range[0]:.0f} – {dist_range[1]:.0f} m")
            print(f"{'-'*80}")
            print(f"{'Torps':<7} | {'Survival (%)':<14} | {'Mean Dist (m)':<22} | {'Pursuit Time (s)':<20}")
            print(f"{'-'*80}")

            for n_torps in torpedo_counts:
                env = AuvEvasionEnv(
                    n_torpedoes    = n_torps,
                    min_spawn_dist = dist_range[0],
                    max_spawn_dist = dist_range[1],
                    mode           = "training",
                )

                survivals          = 0
                terminal_distances = []
                pursuit_times      = []

                for ep in range(episodes_per_scenario):
                    state, _ = env.reset()
                    done = False

                    while not done:
                        # Pure exploitation — matches the target policy exactly
                        action = agent.select_action(state, epsilon=0.0, noise_std=0.0)
                        state, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated

                    if info.get("survived", False):
                        survivals += 1
                        terminal_distances.append(env.current_min_distance)
                    else:
                        pursuit_times.append(env.current_step)

                survival_rate = (survivals / episodes_per_scenario) * 100.0

                mean_dist = np.mean(terminal_distances) if terminal_distances else 0.0
                std_dist  = np.std(terminal_distances)  if terminal_distances else 0.0
                mean_time = np.mean(pursuit_times)      if pursuit_times      else 0.0
                std_time  = np.std(pursuit_times)       if pursuit_times      else 0.0

                dist_str = f"{mean_dist:.0f} ± {std_dist:.0f}"
                time_str = f"{mean_time:.1f} ± {std_time:.1f}"

                print(f"{n_torps:<7} | {survival_rate:>12.2f} % | "
                      f"{dist_str:>20} | {time_str:>18}")

            print()


if __name__ == "__main__":
    evaluate_model()
