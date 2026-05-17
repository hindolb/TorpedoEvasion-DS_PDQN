import numpy as np
import torch
from collections import deque

from Environment_PDQN_DS import AuvEvasionEnv
from PDQN_DS import PDQNAgent


# ------------------------- Exploration schedule -------------------------------
NOISE_START   = 1.0     
NOISE_END     = 0.05
EPSILON_START = 1.0
EPSILON_END   = 0.10
DECAY_EPISODES_PER_LEVEL = 350          

# Episode budget per stage - beyond this the loop advances even without target
MAX_EPISODES_PER_STAGE = 10_000
# Total hard budget across all stages (safety net)
TOTAL_EPISODE_BUDGET   = 60_000

# Warm-start exploration boost when advancing curriculum stages
STAGE_ADVANCE_EPSILON_BUMP = 0.25

# ------------------------ Curriculum (Algorithm 1) ----------------------------
CURRICULUM_STAGES = [
    {"torps": 1, "dist": (1500.0, 2400.0), "target": 90.0},
    {"torps": 2, "dist": (1500.0, 2400.0), "target": 85.0},
    {"torps": 3, "dist": (1500.0, 2400.0), "target": 75.0},
]


def _build_env(stage: dict) -> AuvEvasionEnv:
    """Construct an environment configured for the given curriculum stage."""
    return AuvEvasionEnv(
        n_torpedoes    = stage["torps"],
        min_spawn_dist = stage["dist"][0],
        max_spawn_dist = stage["dist"][1],
    )


def train_auv():
    # ------------------ Initialise stage 1 environment ------------------------
    stage_idx     = 0
    current_stage = CURRICULUM_STAGES[stage_idx]
    env           = _build_env(current_stage)

    # Infer dimensionality from the env's spaces
    feature_dim    = env.observation_space.shape[1]   # 20
    discrete_dim   = env.action_space[0].nvec[0]      # 37
    continuous_dim = env.action_space[1].shape[0]     # 2

    agent = PDQNAgent(
        state_dim    = feature_dim,
        num_discrete = discrete_dim,
        param_dim    = continuous_dim,
    )

    survival_window           = deque(maxlen=100)
    episodes_in_current_level = 0
    total_episodes            = 0
    epsilon_start             = EPSILON_START

    print(f"=== Starting Curriculum ===")
    print(f"  Stage {stage_idx + 1}/{len(CURRICULUM_STAGES)}  |  "
          f"torpedoes = {current_stage['torps']}  |  "
          f"distance = {current_stage['dist']}  |  "
          f"target survival = {current_stage['target']}%")

    training_active = True
    while training_active:

        # --- Episode setup ---------------------------------------------------
        state, _ = env.reset()
        agent.reset_noise()           
        episode_reward = 0.0
        done           = False

        # Linear decay of epsilon and OU sigma within the current stage
        decay_fraction  = min(1.0, episodes_in_current_level / DECAY_EPISODES_PER_LEVEL)
        current_noise   = max(NOISE_END,
                              NOISE_START   - (NOISE_START   - NOISE_END  ) * decay_fraction)
        current_epsilon = max(EPSILON_END,
                              epsilon_start - (epsilon_start - EPSILON_END) * decay_fraction)

        # --- Episode rollout -------------------------------------------------
        while not done:
            action_tuple = agent.select_action(
                state, epsilon=current_epsilon, noise_std=current_noise,
            )
            next_state, reward, terminated, truncated, info = env.step(action_tuple)
            done = terminated or truncated

            discrete_act, continuous_act = action_tuple
            agent.remember(state, discrete_act, continuous_act,
                           reward, next_state, done)
            agent.learn()

            state           = next_state
            episode_reward += reward

        # --- Episode bookkeeping --------------------------------------------
        survived = bool(info.get("survived", False))
        survival_window.append(1 if survived else 0)
        survival_rate = (sum(survival_window) / len(survival_window)) * 100.0

        episodes_in_current_level += 1
        total_episodes            += 1

        if total_episodes % 20 == 0:
            print(f"Ep {total_episodes:05d} | "
                  f"Stage {stage_idx + 1} ({current_stage['torps']} torps) | "
                  f"Reward {episode_reward:>+8.1f} | "
                  f"Survival {survival_rate:>5.1f}% | "
                  f"eps {current_epsilon:.2f} | noise {current_noise:.2f}")

        # --- Curriculum advancement -----------------------------------------
        target_reached  = (len(survival_window) == 100
                           and survival_rate >= current_stage["target"])
        stage_exhausted = episodes_in_current_level >= MAX_EPISODES_PER_STAGE

        if target_reached or stage_exhausted:
            if stage_idx < len(CURRICULUM_STAGES) - 1:
                stage_idx += 1
                current_stage = CURRICULUM_STAGES[stage_idx]

                advance_reason = "target reached" if target_reached else "episode budget"
                print(f"\n{'=' * 70}")
                print(f"  >>> ADVANCING TO STAGE {stage_idx + 1}  ({advance_reason})")
                print(f"      torpedoes = {current_stage['torps']}, "
                      f"distance = {current_stage['dist']}, "
                      f"target = {current_stage['target']}%")
                print(f"{'=' * 70}\n")

                # Re-instantiate env for the new stage
                env = _build_env(current_stage)

                # Clear replay buffer so old-difficulty data does not pollute
                # the new stage's value estimates
                agent.memory.buffer.clear()

                # Bump exploration to help the agent adapt to harder environment
                epsilon_start             = min(1.0, current_epsilon
                                                + STAGE_ADVANCE_EPSILON_BUMP)
                survival_window.clear()
                episodes_in_current_level = 0
            else:
                print("\nAll curriculum stages completed.")
                training_active = False

        if total_episodes >= TOTAL_EPISODE_BUDGET:
            print("\nReached total episode budget. Stopping training.")
            training_active = False

    # --- Save final checkpoint  [MOD-3] -------------------------------------
    checkpoint = {
        "actor_encoder":  agent.actor_encoder.state_dict(),
        "critic_encoder": agent.critic_encoder.state_dict(),
        "param_net":      agent.param_net.state_dict(),
        "q_net":          agent.q_net.state_dict(),
        # Config so inference can rebuild the agent without hardcoding dims
        "state_dim":      feature_dim,
        "num_discrete":   discrete_dim,
        "param_dim":      continuous_dim,
        # Curriculum metadata for later diagnosis
        "final_stage":    stage_idx + 1,
        "final_torps":    current_stage["torps"],
        "total_episodes": total_episodes,
    }
    torch.save(checkpoint, "pdqn_ds_final.pth")

    # Per-network files for backwards compatibility with the original layout
    torch.save(agent.param_net.state_dict(), "pdqn_param_final.pth")
    torch.save(agent.q_net.state_dict(),     "pdqn_q_final.pth")

    print(f"\nTraining complete.")
    print(f"  total episodes      : {total_episodes}")
    print(f"  final torpedo count : {current_stage['torps']}")
    print(f"  checkpoint          : pdqn_ds_final.pth")


if __name__ == "__main__":
    train_auv()
