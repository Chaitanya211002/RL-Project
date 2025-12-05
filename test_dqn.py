import torch
from dqn_agent import DQN, DQNAgent

agent = DQNAgent()
state = torch.randn(8)

q_values = agent.policy(state)
print("Q-values:", q_values)

buffer = agent.replay
buffer.push([0]*8, 1, 0.5, [0]*8, False)
print("Buffer length:", len(buffer))

