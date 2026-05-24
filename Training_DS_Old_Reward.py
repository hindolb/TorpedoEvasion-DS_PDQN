import numpy as np
import torch
from collections import deque

from Environment_PDQN_DS_Old_Reward import AuvEvasionEnv
from PDQN_DS import PDQNAgent


# ------------------------- Exploration schedule -------------------------------
NOISE_START   = 1.0
NOISE_END     = 0.02        
EPSILON_START = 1.0
EPSILON_END   = 0.02        

DECAY_EPISODES_PER_LEVEL = 350

# Episode budget per stage - beyond this the loop advances even without target
MAX_EPISODES_PER_STAGE = 2000
# Total hard budget across all stages (safety net)
TOTAL_EPISODE_BUDGET   = 8000

# Warm-start exploration boost when advancing curriculum stages
STAGE_ADVANCE_EPSILON_BUMP = 0.25

# ------------------------ Curriculum (Algorithm 1) ----------------------------
CURRICULUM_STAGES = [
    {"torps": 1, "dist": (1500.0, 1800.0), "target": 90.0},
    {"torps": 2, "dist": (1500.0, 1800.0), "target": 80.0},
    {"torps": 3, "dist": (1500.0, 1800.0), "target": 70.0},
]

# --- Fine-tuning phase (runs on final stage after curriculum completes) -------
FINE_TUNE_EPISODES = 500
FINE_TUNE_EPSILON  = 0.02
FINE_TUNE_NOISE    = 0.01

SURVIVAL_WINDOW_SIZE = 200   

ADVANCE_PATIENCE = 10


def _build_env(stage: dict) -> AuvEvasionEnv:
    """Construct an environment configured for the given curriculum stage."""
    return AuvEvasionEnv(
        n_torpedoes    = stage["torps"],
        min_spawn_dist = stage["dist"][0],
        max_spawn_dist = stage["dist"][1],
        mode           = "training"
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


    survival_window           = deque(maxlen=SURVIVAL_WINDOW_SIZE)
    reward_window             = deque(maxlen=SURVIVAL_WINDOW_SIZE)
    episodes_in_current_level = 0
    total_episodes            = 0
    epsilon_start             = EPSILON_START
    best_survival_rate        = 0.0
    checks_above_target       = 0   

    print(f"=== Starting Curriculum ===")
    print(f"  Stage {stage_idx + 1}/{len(CURRICULUM_STAGES)}  |  "
          f"torpedoes = {current_stage['torps']}  |  "
          f"distance = {current_stage['dist']}  |  "
          f"target survival = {current_stage['target']}%"
          )

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
        reward_window.append(episode_reward)
        survival_rate = (sum(survival_window) / len(survival_window)) * 100.0
        avg_reward    = sum(reward_window) / len(reward_window)

        episodes_in_current_level += 1
        total_episodes            += 1

        # --- Save best checkpoint per stage ----------------------------------
        if (len(survival_window) == SURVIVAL_WINDOW_SIZE
                and survival_rate > best_survival_rate):
            best_survival_rate = survival_rate
            torch.save({
                "actor_encoder":  agent.actor_encoder.state_dict(),
                "critic_encoder": agent.critic_encoder.state_dict(),
                "param_net":      agent.param_net.state_dict(),
                "q_net":          agent.q_net.state_dict(),
                "state_dim":      feature_dim,
                "num_discrete":   discrete_dim,
                "param_dim":      continuous_dim,
                "stage":          stage_idx + 1,
                "survival_rate":  survival_rate,
                "total_episodes": total_episodes,
            }, f"pdqn_ds_best_stage{stage_idx + 1}.pth")

        # --- Curriculum advancement ------------------------------------------
        exploration_done = (episodes_in_current_level >= DECAY_EPISODES_PER_LEVEL)
        window_full      = (len(survival_window) == SURVIVAL_WINDOW_SIZE)
        stage_exhausted  = (episodes_in_current_level >= MAX_EPISODES_PER_STAGE)

        if total_episodes % 10 == 0 and window_full and exploration_done:
            if survival_rate >= current_stage["target"]:
                checks_above_target += 1
            else:
                checks_above_target = 0

        target_reached  = (checks_above_target >= ADVANCE_PATIENCE)

        if total_episodes % 10 == 0:
            print(f"Ep {total_episodes:05d} | "
                  f"Stage {stage_idx + 1} ({current_stage['torps']} torps) | "
                  f"Reward {episode_reward:>+8.1f} (avg {avg_reward:>+8.1f}) | "
                  f"Survival {survival_rate:>5.1f}% | "
                  f"eps {current_epsilon:.2f} | noise {current_noise:.2f} | "
                  f"patience {checks_above_target}/{ADVANCE_PATIENCE} | "
                  f"best = {best_survival_rate:.1f}%"
                  )

        if target_reached or stage_exhausted:
            advance_reason = "target reached" if target_reached else "episode budget"
            print(f"\nEp {total_episodes:05d} | "
                  f"Stage {stage_idx + 1} COMPLETE ({advance_reason}) | "
                  f"Survival {survival_rate:>5.1f}% | "
                  f"best = {best_survival_rate:.1f}%"
                  )

            if stage_idx < len(CURRICULUM_STAGES) - 1:
                stage_idx    += 1
                current_stage = CURRICULUM_STAGES[stage_idx]

                print(f"\n{'=' * 70}")
                print(f"  >>> ADVANCING TO STAGE {stage_idx + 1}  ({advance_reason})")
                print(f"      torpedoes = {current_stage['torps']}, "
                      f"distance = {current_stage['dist']}, "
                      f"target = {current_stage['target']}%")
                print(f"{'=' * 70}\n")

                env = _build_env(current_stage)
                agent.memory.buffer.clear()

                epsilon_start             = min(1.0, current_epsilon
                                                + STAGE_ADVANCE_EPSILON_BUMP)
                survival_window.clear()
                reward_window.clear()
                episodes_in_current_level = 0
                best_survival_rate        = 0.0
                checks_above_target       = 0   
            else:
                print("\nAll curriculum stages completed.")
                training_active = False

        if total_episodes >= TOTAL_EPISODE_BUDGET:
            print("\nReached total episode budget. Stopping training.")
            training_active = False

    print(f"\n{'=' * 70}")
    print(f"  Fine-tuning: {FINE_TUNE_EPISODES} episodes at "
          f"epsilon={FINE_TUNE_EPSILON}, noise={FINE_TUNE_NOISE}")
    print(f"{'=' * 70}\n")

    ft_survival_window = deque(maxlen=SURVIVAL_WINDOW_SIZE)
    for ft_ep in range(FINE_TUNE_EPISODES):
        state, _ = env.reset()
        agent.reset_noise()
        ft_episode_reward = 0.0
        done = False

        while not done:
            action_tuple = agent.select_action(
                state, epsilon=FINE_TUNE_EPSILON, noise_std=FINE_TUNE_NOISE
            )
            next_state, reward, terminated, truncated, info = env.step(action_tuple)
            done = terminated or truncated

            discrete_act, continuous_act = action_tuple
            agent.remember(state, discrete_act, continuous_act,
                           reward, next_state, done)
            agent.learn()

            state             = next_state
            ft_episode_reward += reward

        ft_survived = bool(info.get("survived", False))
        ft_survival_window.append(1 if ft_survived else 0)
        ft_survival_rate = (sum(ft_survival_window) / len(ft_survival_window)) * 100.0
        total_episodes += 1

        if (ft_ep + 1) % 50 == 0:
            print(f"  Fine-tune ep {ft_ep + 1:4d}/{FINE_TUNE_EPISODES} | "
                  f"Reward {ft_episode_reward:>+8.1f} | "
                  f"Survival {ft_survival_rate:>5.1f}%")

        if (len(ft_survival_window) == SURVIVAL_WINDOW_SIZE
                and ft_survival_rate > best_survival_rate):
            best_survival_rate = ft_survival_rate
            torch.save({
                "actor_encoder":  agent.actor_encoder.state_dict(),
                "critic_encoder": agent.critic_encoder.state_dict(),
                "param_net":      agent.param_net.state_dict(),
                "q_net":          agent.q_net.state_dict(),
                "state_dim":      feature_dim,
                "num_discrete":   discrete_dim,
                "param_dim":      continuous_dim,
                "stage":          "fine_tune",
                "survival_rate":  ft_survival_rate,
                "total_episodes": total_episodes,
            }, "pdqn_ds_best_finetune.pth")

    print(f"\n  Fine-tune complete | "
          f"Final survival rate: {ft_survival_rate:.1f}% | "
          f"Best: {best_survival_rate:.1f}%")

    # --- Save final checkpoint -----------------------------------------------
    checkpoint = {
        "actor_encoder":  agent.actor_encoder.state_dict(),
        "critic_encoder": agent.critic_encoder.state_dict(),
        "param_net":      agent.param_net.state_dict(),
        "q_net":          agent.q_net.state_dict(),
        "state_dim":      feature_dim,
        "num_discrete":   discrete_dim,
        "param_dim":      continuous_dim,
        "final_stage":    stage_idx + 1,
        "final_torps":    current_stage["torps"],
        "total_episodes": total_episodes,
    }
    torch.save(checkpoint, "pdqn_ds_final.pth")
    torch.save(agent.param_net.state_dict(), "pdqn_param_final.pth")
    torch.save(agent.q_net.state_dict(),     "pdqn_q_final.pth")

    print(f"\nTraining complete.")
    print(f"  total episodes      : {total_episodes}")
    print(f"  final torpedo count : {current_stage['torps']}")
    print(f"  checkpoint (final)  : pdqn_ds_final.pth")
    print(f"  checkpoint (best)   : pdqn_ds_best_stage*.pth")
    print(f"  checkpoint (ft best): pdqn_ds_best_finetune.pth")


if __name__ == "__main__":
    train_auv()
