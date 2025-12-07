import numpy as np

from data_pipeline import load_processed_data, train_val_test_split
from trading_env import TradingEnv
from q_learning_agent import QLearningAgent
from state_discretization import discretize_state


def evaluate(agent: QLearningAgent, test_df):
    env = TradingEnv(test_df)
    state = env.reset()

    rewards = []
    portfolios = []

    while True:
        row = env.df.iloc[env.t]
        _, _, _, state_id = discretize_state(row, env.cash, env.shares, env.initial_cash)

        action = int(np.argmax(agent.Q[state_id]))
        next_state, reward, done, info = env.step(action)

        rewards.append(reward)
        portfolios.append(info["portfolio"])

        state = next_state

        if done:
            break

    total_reward = float(np.sum(rewards))
    final_value = float(portfolios[-1]) if portfolios else env.initial_cash
    returns = np.array(rewards)
    sharpe_ratio = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252) if len(returns) > 0 else 0.0

    return total_reward, final_value, sharpe_ratio


if __name__ == "__main__":
    print("Loading processed data...")
    df = load_processed_data()
    _, _, test_df = train_val_test_split(df)

    print("Loading learned Q-table...")
    agent = QLearningAgent()
    agent.load("q_table.npy")
    agent.epsilon = 0.0

    total_reward, final_value, sharpe = evaluate(agent, test_df)

    print("\n=== Q-Learning Evaluation (Test Set) ===")
    print(f"Total Reward: {total_reward:.6f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")

