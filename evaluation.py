import numpy as np
from data_pipeline import load_processed_data, train_val_test_split
from trading_env import TradingEnv
from dqn_agent import DQNAgent
import torch

def evaluate_agent(agent, test_df):
    """
    Run the trained DQN agent on the test dataset.
    Returns total reward, final portfolio value, and Sharpe ratio.
    """

    env = TradingEnv(test_df)
    state = env.reset()
    done = False

    rewards = []
    values = []

    while not done:
        with torch.no_grad():
            q_values = agent.policy(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(q_values).item()

        next_state, reward, done, info = env.step(action)

        rewards.append(reward)
        values.append(info["portfolio"])

        state = next_state

    total_reward = np.sum(rewards)
    final_value = values[-1]

    # Compute daily returns for Sharpe Ratio
    returns = np.array(rewards)
    sharpe_ratio = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)

    return total_reward, final_value, sharpe_ratio


if __name__ == "__main__":
    print("Loading processed data...")
    df = load_processed_data()
    train, val, test = train_val_test_split(df)

    print("Loading trained DQN...")
    agent = DQNAgent()
    agent.policy.load_state_dict(torch.load("dqn_weights.pth"))  # Ensure you saved weights

    total_reward, final_value, sharpe = evaluate_agent(agent, test)

    print("\n=== Test Set Evaluation ===")
    print(f"Total Reward: {total_reward:.6f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Sharpe Ratio: {sharpe:.4f}")
