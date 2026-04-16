import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

#Hyperparameters
LR_PARAM = 1e-5        # Perfect as-is (Parameter networks need slow, stable updates)
LR_Q = 1e-4            # Perfect as-is (Q-network can learn faster)
GAMMA = 0.99           # Perfect as-is (Allows agent to look ~100 steps into the future)
TAU = 0.005            # Perfect as-is (Standard soft-update rate)
BUFFER_SIZE = int(1e5) # Perfect as-is

# CHANGE THIS:
BATCH_SIZE = 128       # Increase from 64 to 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PDQNReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, discrete_action, continuous_params, reward, next_state, done):
        self.buffer.append((state, discrete_action, continuous_params, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, discrete_actions, continuous_params, rewards, next_states, dones = zip(*batch)
        
        # States are now lists of 2D matrices (n_torpedoes, 20). 
        # Wrapping in np.array creates a 3D tensor (BATCH_SIZE, n_torpedoes, 20)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(np.array(discrete_actions)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(continuous_params)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 1. Parameter Network (DeepSets)
# ==========================================
class ParameterNetwork(nn.Module):
    def __init__(self, feature_dim=20, continuous_dim=4):
        super(ParameterNetwork, self).__init__()
        
        # Siamese MLP (Phi): Processes each (AUV+Torpedo) pair independently
        self.phi = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Output MLP (Rho): Processes the summed embeddings
        self.rho = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, continuous_dim)
        )

    def forward(self, state):
        # state shape: (Batch, N_Torpedoes, 20)
        
        # 1. Pass each element through Siamese MLP -> (Batch, N_Torpedoes, 128)
        phi_out = self.phi(state)
        
        # 2. Permutation-invariant aggregation (Sum) -> (Batch, 128)
        agg = torch.sum(phi_out, dim=1)
        
        # 3. Pass through Output MLP -> (Batch, 4)
        params = torch.tanh(self.rho(agg))
        return params

# ==========================================
# 2. Q-Network (DeepSets)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, feature_dim=20, discrete_dim=6, continuous_dim=4):
        super(QNetwork, self).__init__()
        
        # Siamese MLP (Phi)
        self.phi = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        # Output MLP (Rho): Expects the aggregated embedding + the continuous parameters concatenated
        self.rho = nn.Sequential(
            nn.Linear(128 + continuous_dim, 128),
            nn.ReLU(),
            nn.Linear(128, discrete_dim)
        )

    def forward(self, state, continuous_params):
        # state shape: (Batch, N_Torpedoes, 20)
        
        # 1. Pass each element through Siamese MLP -> (Batch, N_Torpedoes, 128)
        phi_out = self.phi(state)
        
        # 2. Permutation-invariant aggregation (Sum) -> (Batch, 128)
        agg = torch.sum(phi_out, dim=1)
        
        # 3. Concatenate global continuous parameters to the aggregated state -> (Batch, 128 + 4)
        x = torch.cat([agg, continuous_params], dim=1)
        
        # 4. Pass through Output MLP -> (Batch, 6)
        q_values = self.rho(x)
        return q_values

# ==========================================
# 3. P-DQN Agent
# ==========================================
class PDQNAgent:
    def __init__(self, feature_dim, discrete_dim=6, continuous_dim=4):
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim
        
        # We now pass feature_dim (20) instead of a flat state_dim
        self.param_net = ParameterNetwork(feature_dim, continuous_dim).to(device)
        self.param_target = ParameterNetwork(feature_dim, continuous_dim).to(device)
        self.param_target.load_state_dict(self.param_net.state_dict())
        self.param_optimizer = optim.Adam(self.param_net.parameters(), lr=LR_PARAM)

        self.q_net = QNetwork(feature_dim, discrete_dim, continuous_dim).to(device)
        self.q_target = QNetwork(feature_dim, discrete_dim, continuous_dim).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=LR_Q)

        self.memory = PDQNReplayBuffer(BUFFER_SIZE)

    def select_action(self, state, epsilon=0.1, noise_std=0.1):
        # state shape coming from env is (N_Torpedoes, 20)
        # unsqueeze(0) converts it to (1, N_Torpedoes, 20) for batch processing
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        self.param_net.eval()
        self.q_net.eval()
        with torch.no_grad():
            continuous_params = self.param_net(state_tensor)
            q_values = self.q_net(state_tensor, continuous_params)
        self.param_net.train()
        self.q_net.train()

        continuous_params = continuous_params.cpu().data.numpy().flatten()
        
        # Continuous Exploration (Gaussian Noise)
        noise = np.random.normal(0, noise_std, size=self.continuous_dim)
        continuous_params = np.clip(continuous_params + noise, -1.0, 1.0)
        
        # Discrete Exploration (Epsilon-Greedy)
        if random.random() < epsilon:
            discrete_action = random.randint(0, self.discrete_dim - 1)
        else:
            discrete_action = torch.argmax(q_values).item()

        return discrete_action, continuous_params

    def remember(self, state, discrete_action, continuous_params, reward, next_state, done):
        self.memory.push(state, discrete_action, continuous_params, reward, next_state, done)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return 

        states, discrete_actions, continuous_params, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # 1. Update Q-Network
        with torch.no_grad():
            next_params = self.param_target(next_states)
            next_q_values = self.q_target(next_states, next_params)
            max_next_q = next_q_values.max(1, keepdim=True)[0]
            target_q = rewards + (GAMMA * max_next_q * (1 - dones))

        current_q_values = self.q_net(states, continuous_params)
        current_q = current_q_values.gather(1, discrete_actions)
        
        q_loss = F.mse_loss(current_q, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        # NEW: Clip Q-Network Gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        # 2. Update Parameter Network
        pred_params = self.param_net(states)
        param_loss = -self.q_net(states, pred_params).sum(dim=1).mean()
        
        self.param_optimizer.zero_grad()
        param_loss.backward()
        # NEW: Clip Parameter Network Gradients
        torch.nn.utils.clip_grad_norm_(self.param_net.parameters(), max_norm=1.0)
        self.param_optimizer.step()

        # 3. Soft Update
        self.soft_update(self.q_net, self.q_target)
        self.soft_update(self.param_net, self.param_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)