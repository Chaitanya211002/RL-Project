import csv
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from data_pipeline import load_processed_data, train_val_test_split
from trading_env import TradingEnv
from q_learning.q_learning_agent import QLearningAgent
from q_learning.state_discretization import discretize_state


def train():
    df = load_processed_data()
    train_df, _, _ = train_val_test_split(df)

    env = TradingEnv(train_df)
    agent = QLearningAgent(alpha=0.1, alpha_decay=0.999, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.998)

    EPISODES = 700
    log_rows = []
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    for ep in range(EPISODES):
        state = env.reset()
        row = env.df.iloc[env.t]
        _, _, _, state_id = discretize_state(row, env.cash, env.shares, env.initial_cash)

        total_reward = 0.0
        step_rewards = []
        negative_steps = 0

        while True:
            action = agent.select_action(state_id)
            next_state, reward, done, info = env.step(action)

            next_row = env.df.iloc[env.t]
            _, _, _, next_state_id = discretize_state(next_row, env.cash, env.shares, env.initial_cash)

            agent.update(state_id, action, reward, next_state_id, done)

            state_id = next_state_id
            state = next_state

            total_reward += reward
            step_rewards.append(reward)
            if reward < 0:
                negative_steps += 1

            if done:
                break

        agent.decay_epsilon()

        steps = len(step_rewards)
        avg_reward = total_reward / steps if steps else 0.0
        min_step = min(step_rewards) if step_rewards else 0.0
        max_step = max(step_rewards) if step_rewards else 0.0

        print(
            f"Episode {ep+1}: Total={total_reward:.4f}, Avg={avg_reward:.6f}, "
            f"Steps={steps}, NegSteps={negative_steps}, "
            f"Range=[{min_step:.6f}, {max_step:.6f}], "
            f"Final=${info['portfolio']:.2f}, Epsilon={agent.epsilon:.4f}"
        )

        log_rows.append(
            {
                "episode": ep + 1,
                "total_reward": total_reward,
                "avg_reward": avg_reward,
                "steps": steps,
                "negative_steps": negative_steps,
                "min_step_reward": min_step,
                "max_step_reward": max_step,
                "final_portfolio": info["portfolio"],
                "epsilon": agent.epsilon,
            }
        )

    q_table_path = os.path.join(logs_dir, "q_table.npy")
    agent.save(q_table_path)
    print(f"Saved Q-table to {q_table_path}")

    training_log_path = os.path.join(logs_dir, "q_learning_training_log.csv")
    with open(training_log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "total_reward",
                "avg_reward",
                "steps",
                "negative_steps",
                "min_step_reward",
                "max_step_reward",
                "final_portfolio",
                "epsilon",
            ],
        )
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Saved training log to {training_log_path}")


if __name__ == "__main__":
    train()

