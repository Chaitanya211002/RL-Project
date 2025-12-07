import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states: int = 27,
        n_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def select_action(self, state_id: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state_id]))

    def update(self, state_id: int, action: int, reward: float, next_state_id: int, done: bool):
        current = self.Q[state_id, action]
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state_id])

        self.Q[state_id, action] = current + self.alpha * (target - current)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        np.save(path, self.Q)

    def load(self, path: str):
        self.Q = np.load(path)

