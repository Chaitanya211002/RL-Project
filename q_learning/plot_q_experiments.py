import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


def plot_learning_curves(log_path: str = os.path.join("logs", "q_experiments_log.csv"), smooth_window: int = 10):
    df = pd.read_csv(log_path)
    configs = df["config"].unique()

    plt.figure(figsize=(10, 6))
    for cfg in configs:
        sub = df[df["config"] == cfg].copy()
        plt.plot(sub["episode"], sub["total_reward"], label=f"{cfg} (raw)", alpha=0.35)
        if smooth_window > 1:
            sub["smoothed"] = sub["total_reward"].rolling(window=smooth_window, min_periods=1).mean()
            plt.plot(sub["episode"], sub["smoothed"], label=f"{cfg} (ema{smooth_window})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Q-Learning Total Reward per Episode (raw + rolling mean {smooth_window})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("logs", "q_learning_total_reward.png"), dpi=200)
    plt.close()
    print("Saved logs/q_learning_total_reward.png")


def plot_summary(summary_path: str = os.path.join("logs", "q_experiments_summary.csv")):
    df = pd.read_csv(summary_path)
    configs = df["config"]

    plt.figure(figsize=(10, 5))
    plt.bar(configs, df["test_final_portfolio"])
    plt.ylabel("Final Portfolio (Test)")
    plt.title("Test Final Portfolio by Config")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join("logs", "q_learning_final_portfolio.png"), dpi=200)
    plt.close()
    print("Saved logs/q_learning_final_portfolio.png")

    plt.figure(figsize=(10, 5))
    plt.bar(configs, df["test_sharpe"])
    plt.ylabel("Sharpe (Test)")
    plt.title("Test Sharpe by Config")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join("logs", "q_learning_sharpe.png"), dpi=200)
    plt.close()
    print("Saved logs/q_learning_sharpe.png")


def plot_eval_sharpe(eval_path: str = os.path.join("logs", "q_experiments_eval.csv")):
    df = pd.read_csv(eval_path)
    configs = df["config"].unique()

    plt.figure(figsize=(10, 6))
    for cfg in configs:
        sub = df[df["config"] == cfg]
        plt.plot(sub["episode"], sub["eval_sharpe"], label=cfg)
    plt.xlabel("Episode (eval checkpoints)")
    plt.ylabel("Sharpe (greedy eval)")
    plt.title("Q-Learning Sharpe over Episodes (periodic eval)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join("logs", "q_learning_eval_sharpe.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_learning_curves()
    plot_summary()
    plot_eval_sharpe()

