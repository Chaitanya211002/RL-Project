import csv
import os
import sys
import random
from dataclasses import dataclass, asdict
from typing import List

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from data_pipeline import load_processed_data, train_val_test_split
from trading_env import TradingEnv
from q_learning.q_learning_agent import QLearningAgent
from q_learning.state_discretization import discretize_state
from q_learning.evaluate_q_learning import evaluate


@dataclass
class QExperimentConfig:
    name: str
    episodes: int = 200
    seed: int = 0
    alpha: float = 0.05
    alpha_decay: float = 0.999
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    final_greedy_at: float = 0.0  # 0 means no forced greedy; >0 forces epsilon=0 after that fraction
    eval_every: int = 1           # evaluate every N episodes


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def train_single(cfg: QExperimentConfig, train_df, eval_df):
    set_seed(cfg.seed)

    env = TradingEnv(train_df)
    agent = QLearningAgent(
        alpha=cfg.alpha,
        alpha_decay=cfg.alpha_decay,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon,
        epsilon_min=cfg.epsilon_min,
        epsilon_decay=cfg.epsilon_decay,
    )

    log_rows = []
    eval_rows = []

    for ep in range(cfg.episodes):
        state = env.reset()
        row = env.df.iloc[env.t]
        _, _, _, state_id = discretize_state(row, env.cash, env.shares, env.initial_cash)

        total_reward = 0.0
        step_rewards: List[float] = []
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
        if cfg.final_greedy_at > 0 and (ep + 1) >= int(cfg.final_greedy_at * cfg.episodes):
            agent.epsilon = 0.0  # force greedy in final phase

        steps = len(step_rewards)
        avg_reward = total_reward / steps if steps else 0.0
        min_step = min(step_rewards) if step_rewards else 0.0
        max_step = max(step_rewards) if step_rewards else 0.0

        # Console log per episode
        print(
            f"[{cfg.name}] Ep {ep+1}/{cfg.episodes} | "
            f"Total={total_reward:.4f}, Avg={avg_reward:.6f}, "
            f"Steps={steps}, NegSteps={negative_steps}, "
            f"Range=[{min_step:.6f}, {max_step:.6f}], "
            f"Final=${info['portfolio']:.2f}, Eps={agent.epsilon:.4f}"
        )

        log_rows.append(
            {
                "config": cfg.name,
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

        # periodic greedy evaluation on eval_df (default: test)
        if (ep + 1) % cfg.eval_every == 0:
            eval_reward, eval_value, eval_sharpe = evaluate(agent, eval_df)
            eval_rows.append(
                {
                    "config": cfg.name,
                    "episode": ep + 1,
                    "eval_total_reward": eval_reward,
                    "eval_final_portfolio": eval_value,
                    "eval_sharpe": eval_sharpe,
                }
            )

    return agent, log_rows, eval_rows


def main():
    df = load_processed_data()
    train_df, _, test_df = train_val_test_split(df)
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    experiments = [
        # Slow-decay family (best observed so far)
        QExperimentConfig(name="fast_decay", episodes=200, seed=0, epsilon=1, epsilon_decay=0.98),
        #QExperimentConfig(name="slow_decay_lowemin", episodes=100, seed=0, epsilon=1, epsilon_decay=0.999, epsilon_min=0.001),
        #QExperimentConfig(name="slow_decay_faster", episodes=100, seed=0, epsilon=1, epsilon_decay=0.997),
        #QExperimentConfig(name="slow_decay_seed1", episodes=400, seed=1, epsilon=0.7, epsilon_decay=0.999),
        #QExperimentConfig(name="slow_decay_seed2", episodes=400, seed=2, epsilon=0.7, epsilon_decay=0.999),
        QExperimentConfig(name="slow_decay_alphafast", episodes=200, seed=443, epsilon=0.5, epsilon_decay=0.999, alpha_decay=0.997),
        #QExperimentConfig(name="slow_decay_alphafast_443", episodes=100, seed=443, epsilon=0.7, epsilon_decay=0.999, alpha_decay=0.997),
    ]

    all_logs = []
    summary_rows = []
    eval_logs = []

    for cfg in experiments:
        print(f"\n=== Running config: {cfg.name} ===")
        agent, logs, eval_rows = train_single(cfg, train_df, test_df)
        all_logs.extend(logs)
        eval_logs.extend(eval_rows)

        # Greedy test eval using current epsilon schedule result
        total_reward, final_value, sharpe = evaluate(agent, test_df)

        # Strict greedy eval (epsilon = 0) after training
        agent.epsilon = 0.0
        greedy_reward, greedy_value, greedy_sharpe = evaluate(agent, test_df)

        summary_rows.append(
            {
                "config": cfg.name,
                "seed": cfg.seed,
                "episodes": cfg.episodes,
                "alpha": cfg.alpha,
                "alpha_decay": cfg.alpha_decay,
                "gamma": cfg.gamma,
                "epsilon": cfg.epsilon,
                "epsilon_min": cfg.epsilon_min,
                "epsilon_decay": cfg.epsilon_decay,
                "final_greedy_at": cfg.final_greedy_at,
                "eval_every": cfg.eval_every,
                "test_total_reward": total_reward,
                "test_final_portfolio": final_value,
                "test_sharpe": sharpe,
                "test_total_reward_greedy": greedy_reward,
                "test_final_portfolio_greedy": greedy_value,
                "test_sharpe_greedy": greedy_sharpe,
            }
        )

    per_episode_path = os.path.join(logs_dir, "q_experiments_log.csv")
    with open(per_episode_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
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
        writer.writerows(all_logs)
    print(f"Saved per-episode logs to {per_episode_path}")

    summary_path = os.path.join(logs_dir, "q_experiments_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "seed",
                "episodes",
                "alpha",
                "alpha_decay",
                "gamma",
                "epsilon",
                "epsilon_min",
                "epsilon_decay",
                "final_greedy_at",
                "eval_every",
                "test_total_reward",
                "test_final_portfolio",
                "test_sharpe",
                "test_total_reward_greedy",
                "test_final_portfolio_greedy",
                "test_sharpe_greedy",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary metrics to {summary_path}")

    eval_path = os.path.join(logs_dir, "q_experiments_eval.csv")
    with open(eval_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "episode",
                "eval_total_reward",
                "eval_final_portfolio",
                "eval_sharpe",
            ],
        )
        writer.writeheader()
        writer.writerows(eval_logs)
    print(f"Saved periodic evals to {eval_path}")


if __name__ == "__main__":
    main()

