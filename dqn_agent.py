import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim=8, output_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch=64):
        batch = random.sample(self.buffer, batch)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.tensor(s, dtype=torch.float32),
            torch.tensor(a),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(ns, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim=8, action_dim=3, gamma=0.99, lr=0.001):
        self.gamma = gamma

        self.policy = DQN(state_dim, action_dim)
        self.target = DQN(state_dim, action_dim)
        self.target.load_state_dict(self.policy.state_dict())

        self.optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.replay = ReplayBuffer()

        self.epsilon = 1.0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        state = torch.tensor(state, dtype=torch.float32)
        q = self.policy(state)
        return int(torch.argmax(q).item())

    def update(self, batch_size=64):
        if len(self.replay) < batch_size:
            return

        s, a, r, ns, d = self.replay.sample(batch_size)

        q_vals = self.policy(s).gather(1, a.unsqueeze(1)).squeeze()
        next_q = self.target(ns).max(dim=1)[0]
        target = r + self.gamma * next_q * (1 - d)

        loss = nn.MSELoss()(q_vals, target.detach())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()