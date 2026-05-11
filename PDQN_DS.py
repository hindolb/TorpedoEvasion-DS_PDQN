import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

# =============================================================================
# 1. Deep Sets Encoder  (unchanged - architecture matches paper Section III-C)
# =============================================================================

class DeepSetEncoder(nn.Module):
    """
    Processes a variable number of torpedo states concatenated with the AUV
    state. Achieves permutation invariance and cardinality invariance via the
    phi -> sum-pool -> rho decomposition (Paper Eq. 6).

    Input  shape: (batch, n_torpedoes, state_dim)
    Output shape: (batch, latent_dim)
    """
    def __init__(self, state_dim: int = 20, latent_dim: int = 128):
        super().__init__()

        # phi  - Siamese MLP applied identically to every combined state vector
        self.phi = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )

        # rho  - global reasoning MLP applied to the aggregated embedding
        self.rho = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_torpedoes, state_dim = x.size()

        # Apply phi to every element independently (Siamese weights)
        phi_out = self.phi(x.view(-1, state_dim))               # (B*N, latent)
        phi_out = phi_out.view(batch_size, n_torpedoes, -1)     # (B, N, latent)

        # Sum-pool (commutative aggregation — permutation invariant)
        pooled = torch.sum(phi_out, dim=1)                      # (B, latent)

        return self.rho(pooled)                                  # (B, latent)


# =============================================================================
# 2. Actor Network - Deterministic Policy / Parameter Network (DPN)
# =============================================================================

class ParamNet(nn.Module):
    """
    Deterministic Policy / Parameter Network (DPN).

    FIX [CRITICAL-2]: Paper Eq. 3 requires a separate continuous parameter
    vector x_k for EACH of the K=37 primary discrete actions (decoy IDs).
    This allows the policy to learn action-conditional kinematics.

    Output shape: (batch, num_discrete, param_dim)
        - num_discrete = K = 37  (one entry per primary decoy ID d1)
        - param_dim    = 2       (speed, heading)
    The Tanh activation bounds all outputs to [-1, 1].
    """
    def __init__(self, latent_dim: int = 128,
                 param_dim: int = 2,
                 num_discrete: int = 37):
        super().__init__()
        self.num_discrete = num_discrete
        self.param_dim    = param_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_discrete * param_dim),
            nn.Tanh(),   # all outputs in [-1, 1]
        )

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        # Returns (batch, num_discrete, param_dim)
        out = self.fc(state_embedding)
        return out.view(out.shape[0], self.num_discrete, self.param_dim)


# =============================================================================
# 3. Critic Network — Q-Value Network (QVN)
# =============================================================================

class QNet(nn.Module):
    """
    Q-Value Network (QVN).

    FIX [CRITICAL-2]: Given a state embedding and the continuous params x_{d1}
    associated with a specific primary discrete action d1, evaluates Q over all
    secondary discrete actions (d2). This matches Paper Eq. 4 where each
    discrete action k has its own x_k.

    Input:  state_embedding (batch, latent_dim)  +  params (batch, param_dim)
    Output: Q values        (batch, num_secondary)   — one per d2 choice
    """
    def __init__(self, latent_dim: int = 128,
                 param_dim: int = 2,
                 num_secondary: int = 37):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + param_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_secondary),
        )

    def forward(self, state_embedding: torch.Tensor,
                params: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([state_embedding, params], dim=1))


# =============================================================================
# 4. Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Stores transitions (s, d1, d2, x_{d1}, r, s', done).
    d1 and d2 are stored separately so the critic can gather Q(s, d2 | x_{d1})
    without re-encoding the flat joint index.
    """
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, d1: int, d2: int, param_action,
             reward: float, next_state, done: bool):
        self.buffer.append((state, d1, d2, param_action,
                            reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, d1s, d2s, param_actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(d1s),
                np.array(d2s),
                np.array(param_actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones))

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


# =============================================================================
# 5. Ornstein-Uhlenbeck Noise
# =============================================================================

class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise on
    the continuous action (speed, heading).

    FIX [MINOR-1]: reset() must be called at the start of every episode.
    Training_DS.py should add:
        agent.reset_noise()
    immediately after env.reset() in the episode loop.
    """
    def __init__(self, size: int, mu: float = 0.0,
                 theta: float = 0.15, sigma: float = 0.2):
        self.size  = size
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        """Reset noise state to mean. Call at the start of each episode."""
        self.state = np.copy(self.mu)

    def sample(self) -> np.ndarray:
        x          = self.state
        dx         = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class MemoryWrapper:
    def __init__(self, capacity: int = 100_000):
        self.buffer = ReplayBuffer(capacity)


# =============================================================================
# 6. PDQN Agent (Deep Sets + Per-Action Params)
# =============================================================================

class PDQNAgent:
    """
    P-DQN agent with Deep Sets encoder and per-action continuous parameters.

    Architecture matches the paper:
      - Separate DeepSet encoders for actor and critic (Fig. 2).
      - ParamNet outputs K=37 (speed, heading) vectors — one per d1 (Eq. 3).
      - QNet evaluates Q over d2 given d1's specific params (Eq. 4).
      - Actor loss sums Q across all K discrete actions (Eq. 17).
      - Critic loss is MSBE using Bellman target (Eq. 15, 16).
      - Soft target network updates for both encoder pairs (Algorithm 1).

    Hyperparameters match Table I of the paper.
    """
    def __init__(self,
                 state_dim: int   = 20,
                 num_discrete: int = 37,
                 param_dim: int    = 2,
                 # [MODERATE-2] Corrected learning rates from Table I
                 lr_actor:  float  = 1.2527147773442612e-05,
                 lr_critic: float  = 2.4709544319378277e-05,
                 # [MODERATE-2] Corrected gamma and tau from Table I
                 gamma: float      = 0.99,
                 tau:   float      = 0.00194,
                 device: str       = "cuda" if torch.cuda.is_available() else "cpu"):

        # Force CPU — GPU support can be re-enabled by removing this line
        device = "cpu"
        self.device       = device
        self.gamma        = gamma
        self.tau          = tau
        self.num_discrete = num_discrete   # K = 37
        self.param_dim    = param_dim      # 2  (speed, heading)

        # ------------------------------------------------------------------
        # [CRITICAL-1] Separate Deep Sets encoders for actor and critic
        # ------------------------------------------------------------------
        self.actor_encoder        = DeepSetEncoder(state_dim).to(device)
        self.target_actor_encoder = DeepSetEncoder(state_dim).to(device)
        self.target_actor_encoder.load_state_dict(self.actor_encoder.state_dict())

        self.critic_encoder        = DeepSetEncoder(state_dim).to(device)
        self.target_critic_encoder = DeepSetEncoder(state_dim).to(device)
        self.target_critic_encoder.load_state_dict(self.critic_encoder.state_dict())

        # ------------------------------------------------------------------
        # [CRITICAL-2] ParamNet: K separate param vectors (Eq. 3)
        # ------------------------------------------------------------------
        self.param_net        = ParamNet(param_dim=param_dim,
                                         num_discrete=num_discrete).to(device)
        self.target_param_net = ParamNet(param_dim=param_dim,
                                         num_discrete=num_discrete).to(device)
        self.target_param_net.load_state_dict(self.param_net.state_dict())

        # ------------------------------------------------------------------
        # [CRITICAL-2] QNet: Q over d2 given d1's specific params (Eq. 4)
        # ------------------------------------------------------------------
        self.q_net        = QNet(param_dim=param_dim,
                                  num_secondary=num_discrete).to(device)
        self.target_q_net = QNet(param_dim=param_dim,
                                  num_secondary=num_discrete).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # ------------------------------------------------------------------
        # [CRITICAL-1] Disjoint optimisers — actor loss cannot touch critic
        #              weights and vice-versa.
        # ------------------------------------------------------------------
        self.actor_optimizer = optim.Adam(
            list(self.actor_encoder.parameters()) +
            list(self.param_net.parameters()),
            lr=lr_actor,
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic_encoder.parameters()) +
            list(self.q_net.parameters()),
            lr=lr_critic,
        )

        self.memory = MemoryWrapper()
        self.noise  = OUNoise(size=param_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_noise(self):
        """
        [MINOR-1] Reset OU noise to its mean.
        Must be called at the start of every episode in Training_DS.py:
            agent.reset_noise()  # after env.reset()
        """
        self.noise.reset()

    def select_action(self, state: np.ndarray,
                      epsilon: float = 0.0,
                      noise_std: float = 0.0) -> tuple:
        """
        Select a parameterised action (d1, d2, x_{d1}) for the given state.

        Exploration:
          - epsilon-greedy over the joint (d1 x d2) discrete space.
          - OUNoise applied to all K param vectors with the same noise draw
            (one OU sample broadcast over (K, param_dim) — tractable and
             consistent with the single continuous-action noise intent of OU).

        Returns:
            (np.array([d1, d2]),  chosen_params)   <- matches Training_DS.py
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Actor path: K param vectors for this state
            emb_actor  = self.actor_encoder(state_tensor)           # (1, latent)
            params_all = self.param_net(emb_actor)                  # (1, K, 2)

            # Critic path: evaluate Q[d1, d2] matrix
            emb_critic = self.critic_encoder(state_tensor)          # (1, latent)
            # Expand critic embedding to match K copies for the K d1 values
            emb_c_exp  = emb_critic.expand(self.num_discrete, -1)  # (K, latent)
            params_exp = params_all.squeeze(0)                      # (K, 2)

            # Apply exploration noise (broadcast single OU draw to all K rows)
            params_np = params_exp.cpu().numpy().copy()             # (K, 2)
            params_np += noise_std * self.noise.sample()            # broadcast (2,) -> (K, 2)
            params_np  = np.clip(params_np, -1.0, 1.0)
            params_t   = torch.FloatTensor(params_np).to(self.device)

            # Q matrix: q_mat[d1, d2]  (Paper Eq. 5)
            q_mat = self.q_net(emb_c_exp, params_t)                # (K, K)

        if np.random.rand() < epsilon:
            d1 = np.random.randint(self.num_discrete)
            d2 = np.random.randint(self.num_discrete)
        else:
            # a* = argmax_{d1,d2} Q(s, d1, d2, x_{d1})   (Eq. 5)
            flat_idx = q_mat.reshape(-1).argmax().item()
            d1 = flat_idx // self.num_discrete
            d2 = flat_idx % self.num_discrete

        chosen_params = params_np[d1]                               # (2,) for chosen d1
        return (np.array([d1, d2]), chosen_params)

    def remember(self, state, discrete_action, param_action,
                 reward: float, next_state, done: bool):
        """
        Store a transition in the replay buffer.
        discrete_action is np.array([d1, d2]) as returned by select_action.
        param_action is the 2-dim continuous params for the chosen d1.
        """
        d1, d2 = discrete_action
        self.memory.buffer.push(state, int(d1), int(d2),
                                param_action, reward, next_state, done)

    def learn(self, batch_size: int = 256):  # [MODERATE-2] 64 -> 256 (Table I)
        """
        One gradient step for both critic and actor networks.
        Called every environment step during training.
        """
        if len(self.memory.buffer) < batch_size:
            return

        (states, d1s, d2s, param_actions,
         rewards, next_states, dones) = self.memory.buffer.sample(batch_size)

        B = states.shape[0]  # batch size

        # Convert to tensors
        states        = torch.FloatTensor(states).to(self.device)
        next_states   = torch.FloatTensor(next_states).to(self.device)
        d1s           = torch.LongTensor(d1s).to(self.device)           # (B,)
        d2s           = torch.LongTensor(d2s).to(self.device)           # (B,)
        param_actions = torch.FloatTensor(param_actions).to(self.device) # (B, 2)
        rewards       = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones         = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ==================================================================
        # Critic Update  -  MSBE loss  (Paper Eq. 15, 16)
        # ==================================================================
        with torch.no_grad():
            # Target actor: K param vectors for next states
            next_emb_actor  = self.target_actor_encoder(next_states)  # (B, latent)
            next_params_all = self.target_param_net(next_emb_actor)    # (B, K, 2)

            # Target critic: evaluate Q over all (d1, d2) pairs in next state
            # Expand: (B, latent) -> (B, K, latent) -> (B*K, latent)
            next_emb_critic = self.target_critic_encoder(next_states)  # (B, latent)
            next_emb_exp    = (next_emb_critic
                               .unsqueeze(1)
                               .expand(-1, self.num_discrete, -1)
                               .reshape(B * self.num_discrete, -1))    # (B*K, latent)
            next_params_exp = next_params_all.reshape(
                B * self.num_discrete, self.param_dim)                  # (B*K, 2)

            next_q = self.target_q_net(next_emb_exp, next_params_exp)  # (B*K, K)
            next_q = next_q.reshape(B, self.num_discrete,
                                    self.num_discrete)                  # (B, d1, d2)

            # Bellman target: max over the full joint (d1, d2) action space
            max_next_q = next_q.reshape(B, -1).max(1, keepdim=True)[0] # (B, 1)
            target_q   = rewards + (1.0 - dones) * self.gamma * max_next_q

        # Current Q for the stored (d1, x_{d1}, d2) transition
        emb_critic = self.critic_encoder(states)
        q_values   = self.q_net(emb_critic, param_actions)   # (B, K) — over d2
        q_expected = q_values.gather(1, d2s.unsqueeze(1))    # (B, 1)

        critic_loss = F.mse_loss(q_expected, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ==================================================================
        # Actor Update  —  Eq. 17
        # L_μ(ω) = -E_s[ Σ_{k=1}^{K} Q(s, k, μ_ω(s)[k]) ]
        # ==================================================================
        emb_actor  = self.actor_encoder(states)               # (B, latent)
        params_all = self.param_net(emb_actor)                # (B, K, 2) — gradients flow

        # Critic embedding detached: critic weights must not be updated by
        # the actor loss  [CRITICAL-1]
        emb_crit_det = self.critic_encoder(states).detach()   # (B, latent)
        emb_exp = (emb_crit_det
                   .unsqueeze(1)
                   .expand(-1, self.num_discrete, -1)
                   .reshape(B * self.num_discrete, -1))        # (B*K, latent)
        params_exp = params_all.reshape(
            B * self.num_discrete, self.param_dim)             # (B*K, 2)

        # [MODERATE-1] Freeze q_net weights during actor backward pass to
        # prevent ghost gradient accumulation in q_net.fc.
        for p in self.q_net.parameters():
            p.requires_grad_(False)

        q_all = self.q_net(emb_exp, params_exp)               # (B*K, K)
        q_all = q_all.reshape(B, self.num_discrete,
                               self.num_discrete)              # (B, d1, d2)

        # Eq. 17: maximise sum of Q values across all K discrete actions;
        # each action k contributes Q(s, k, μ_ω(s)[k]).
        actor_loss = -q_all.sum(dim=[1, 2]).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Restore q_net gradients for the next critic update
        for p in self.q_net.parameters():
            p.requires_grad_(True)

        # ==================================================================
        # Soft Target Updates  (Algorithm 1, line 25)
        # ==================================================================
        self._soft_update(self.actor_encoder,  self.target_actor_encoder)
        self._soft_update(self.critic_encoder, self.target_critic_encoder)
        self._soft_update(self.param_net,      self.target_param_net)
        self._soft_update(self.q_net,          self.target_q_net)

    def _soft_update(self, local_model: nn.Module,
                     target_model: nn.Module) -> None:
        for t_p, l_p in zip(target_model.parameters(),
                             local_model.parameters()):
            t_p.data.copy_(
                self.tau * l_p.data + (1.0 - self.tau) * t_p.data)
