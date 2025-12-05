from data_pipeline import load_processed_data, train_val_test_split
from trading_env import TradingEnv
from dqn_agent import DQNAgent
from baselines_and_metrics import *

import numpy as np
import matplotlib.pyplot as plt

df = load_processed_data()
train, val, test = train_val_test_split(df)

env = TradingEnv(train)
agent = DQNAgent()

EPISODES = 100

# Store rewards for plotting
episode_rewards = []
episode_portfolios = []

for ep in range(EPISODES):
    state = env.reset()
    total_reward = 0
    steps = 0
    step_rewards = []

    while True:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        agent.replay.push(state, action, reward, next_state, done)
        agent.update()

        state = next_state
        total_reward += reward
        step_rewards.append(reward)
        steps += 1

        if done:
            break

    agent.epsilon = max(0.05, agent.epsilon * 0.98)
    
    # Store metrics
    episode_rewards.append(total_reward)
    episode_portfolios.append(info['portfolio'])
    
    # Calculate statistics
    avg_reward = total_reward / steps
    min_step = min(step_rewards)
    max_step = max(step_rewards)
    negative_steps = sum(1 for r in step_rewards if r < 0)
    
    print(f"Episode {ep+1}: Total={total_reward:.4f}, Avg={avg_reward:.6f}, "
          f"Steps={steps}, NegSteps={negative_steps}, "
          f"Range=[{min_step:.6f}, {max_step:.6f}], "
          f"Final=${info['portfolio']:.2f}")

print("Training complete.")

torch.save(agent.policy.state_dict(), "dqn_weights.pth")
print("Saved trained model to dqn_weights.pth")
