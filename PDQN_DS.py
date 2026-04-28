import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

# ==========================================
# 1. Deep Sets Architecture 
# ==========================================

class DeepSetEncoder(nn.Module):
    """
    Processes variable number of torpedo states combined with AUV state.
    Achieves permutation and cardinality invariance.
    """
    def __init__(self, state_dim=20, latent_dim=128):
        super(DeepSetEncoder, self).__init__()
        
        # Siamese MLP (phi): Applied to each combined state individually
        self.phi = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        
        # Global processing MLP (rho): Applied to the aggregated embedding
        self.rho = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: (batch_size, n_torpedoes, state_dim)
        batch_size, n_torpedoes, state_dim = x.size()
        
        # Flatten to apply Siamese network
        x_flat = x.view(-1, state_dim)
        phi_out = self.phi(x_flat)
        phi_out = phi_out.view(batch_size, n_torpedoes, -1)
        
        # Aggregation: Summation pool
        sum_pooled = torch.sum(phi_out, dim=1)
        
        # Process the aggregated state to produce one unified latent embedding
        out = self.rho(sum_pooled)
        return out


# ==========================================
# 2. Actor-Critic Networks for PDQN
# ==========================================

class ParamNet(nn.Module):
    """
    Deterministic Policy/Parameter Network (DPN)
    Outputs the continuous parameters (speed and heading) for the actions.
    """
    def __init__(self, latent_dim=128, param_dim=2):
        super(ParamNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, param_dim),
            nn.Tanh() # Continuous actions are scaled between -1 and 1
        )
        
    def forward(self, state_embedding):
        return self.fc(state_embedding)


class QNet(nn.Module):
    """
    Q-Value Network (QVN)
    Evaluates expected cumulative rewards for discrete actions based on 
    the state and selected continuous parameters.
    """
    def __init__(self, latent_dim=128, param_dim=2, num_discrete_actions=1369): 
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + param_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_discrete_actions)
        )
        
    def forward(self, state_embedding, params):
        x = torch.cat([state_embedding, params], dim=1)
        return self.fc(x)


# ==========================================
# 3. Buffer and Noise Utils
# ==========================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, discrete_action, param_action, reward, next_state, done):
        self.buffer.append((state, discrete_action, param_action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, discrete_actions, param_actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(discrete_actions), np.array(param_actions), np.array(rewards), np.array(next_states), np.array(dones)
        
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

class OUNoise:
    """Ornstein-Uhlenbeck process for continuous action exploration."""
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class MemoryWrapper:
    def __init__(self, capacity=100000):
        self.buffer = ReplayBuffer(capacity)


# ==========================================
# 4. Deep Sets PDQN Agent
# ==========================================

class PDQNAgent:
    # Match argument order from Training_DS.py: feature_dim, discrete_dim, continuous_dim
    def __init__(self, state_dim=20, num_discrete=37, param_dim=2, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005, device="cuda" if torch.cuda.is_available() else "cpu"):
        device = "cpu"
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.num_discrete = num_discrete
        self.total_discrete = num_discrete * num_discrete # e.g., 37x37 = 1369 combinations
        
        # Deep Sets Encoders
        self.encoder = DeepSetEncoder(state_dim).to(device)
        self.target_encoder = DeepSetEncoder(state_dim).to(device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        
        # Actor Networks (Continuous parameters)
        self.param_net = ParamNet(param_dim=param_dim).to(device)
        self.target_param_net = ParamNet(param_dim=param_dim).to(device)
        self.target_param_net.load_state_dict(self.param_net.state_dict())
        
        # Critic Networks (Discrete Q-Values)
        self.q_net = QNet(param_dim=param_dim, num_discrete_actions=self.total_discrete).to(device)
        self.target_q_net = QNet(param_dim=param_dim, num_discrete_actions=self.total_discrete).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.param_net.parameters()), lr=lr_actor)
        self.critic_optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.q_net.parameters()), lr=lr_critic)
        
        self.memory = MemoryWrapper()
        self.noise = OUNoise(size=param_dim)
        
    def select_action(self, state, epsilon=0.0, noise_std=0.0):
        # Convert state numpy array -> tensor, add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.encoder(state_tensor)
            params = self.param_net(embedding)
            q_values = self.q_net(embedding, params)
        
        # Continuous action mapping (extracting parameters, applying noise)
        params = params.cpu().numpy()[0]
        params += noise_std * self.noise.sample()
        params = np.clip(params, -1.0, 1.0)
        
        # Epsilon-greedy discrete action selection
        if np.random.rand() < epsilon:
            discrete_action = np.random.randint(self.total_discrete)
        else:
            discrete_action = q_values.argmax(dim=1).item()
            
        # Decode 1D action integer back into 2 MultiDiscrete Decoy IDs
        d1 = discrete_action // self.num_discrete
        d2 = discrete_action % self.num_discrete
        
        # Returns tuple expected by Training_DS.py
        return (np.array([d1, d2]), params)
        
    def remember(self, state, discrete_action, param_action, reward, next_state, done):
        # discrete_action is the np.array([d1, d2]) from select_action
        d1, d2 = discrete_action
        
        # Encode back into a flat discrete dimension for Q-Learning Target logic
        flat_discrete_action = d1 * self.num_discrete + d2 
        
        self.memory.buffer.push(state, flat_discrete_action, param_action, reward, next_state, done)
        
    def learn(self, batch_size=64):
        if len(self.memory.buffer) < batch_size:
            return
            
        states, discrete_actions, param_actions, rewards, next_states, dones = self.memory.buffer.sample(batch_size)
        
        # Prepare Batch Tensors
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device) # <-- ADD THIS LINE
        discrete_actions = torch.LongTensor(discrete_actions).to(self.device)
        param_actions = torch.FloatTensor(param_actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # -----------------------------
        # Critic Update
        # -----------------------------
        with torch.no_grad():
            next_emb = self.target_encoder(next_states)
            next_params = self.target_param_net(next_emb)
            next_q = self.target_q_net(next_emb, next_params)
            max_next_q = next_q.max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
        emb = self.encoder(states)
        q_values = self.q_net(emb, param_actions)
        
        # Gather the specific Q-value for the action that was taken
        q_expected = q_values.gather(1, discrete_actions.unsqueeze(1))
        
        critic_loss = F.mse_loss(q_expected, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # -----------------------------
        # Actor Update
        # -----------------------------
        emb_actor = self.encoder(states)
        params = self.param_net(emb_actor)
        q_actor = self.q_net(emb_actor, params)
        
        # To maximize Q-values across the batch, negative sum is used as loss
        actor_loss = -q_actor.sum(dim=1).mean() 
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # -----------------------------
        # Soft Update Targets
        # -----------------------------
        self._soft_update(self.encoder, self.target_encoder)
        self._soft_update(self.param_net, self.target_param_net)
        self._soft_update(self.q_net, self.target_q_net)
        
    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)