import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Mock yfinance to avoid import error if not installed
from unittest.mock import MagicMock
sys.modules["yfinance"] = MagicMock()

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not found, using Matplotlib defaults.")

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from data_pipeline import load_processed_data, train_val_test_split
from trading_env import TradingEnv
from q_learning.state_discretization import discretize_state
from q_learning.q_learning_agent import QLearningAgent

# Setup output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "presentation_assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('ggplot')

def load_data_and_agent():
    print("Loading data...")
    df = load_processed_data()
    train_df, val_df, test_df = train_val_test_split(df)
    
    # Load Q-table
    q_table_path = os.path.join(os.path.dirname(__file__), "logs", "q_table.npy")
    if not os.path.exists(q_table_path):
        raise FileNotFoundError(f"Q-table not found at {q_table_path}")
    
    q_table = np.load(q_table_path)
    
    # Initialize Agent with loaded table
    agent = QLearningAgent()
    agent.Q = q_table
    agent.epsilon = 0.0 # Greedy for evaluation
    
    return df, test_df, agent, q_table

def plot_learning_curve():
    print("Generating Learning Curve...")
    log_path = os.path.join(os.path.dirname(__file__), "logs", "q_learning_training_log.csv")
    if not os.path.exists(log_path):
        print("Warning: Training log not found. Skipping learning curve.")
        return

    df = pd.read_csv(log_path)
    
    # 1. Total Reward vs Episode
    plt.figure(figsize=(10, 6))
    plt.plot(df['episode'], df['total_reward'], label='Raw Reward', alpha=0.3, color='gray')
    
    # Rolling Mean
    window = 20
    df['ma'] = df['total_reward'].rolling(window=window, min_periods=1).mean()
    plt.plot(df['episode'], df['ma'], label=f'{window}-Episode Moving Avg', color='blue', linewidth=2)
    
    plt.title('Agent Learning Progress: Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Portfolio % Change)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_learning_curve.png"), dpi=150)
    plt.close()

    # 2. Epsilon Decay
    plt.figure(figsize=(8, 4))
    plt.plot(df['episode'], df['epsilon'], color='green', linewidth=2)
    plt.title('Exploration vs Exploitation: Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon (Exploration Rate)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_epsilon_decay.png"), dpi=150)
    plt.close()

def plot_policy_heatmaps(q_table):
    print("Generating Policy Heatmaps...")
    
    # State mapping reminders:
    # state_id = price_bin * 9 + rsi_bin * 3 + holding_bin
    # Price(Trend): 0=Down, 1=Flat, 2=Up
    # RSI: 0=Oversold(<30), 1=Neutral, 2=Overbought(>70)
    # Holding: 0=Empty, 1=Partial, 2=Full
    
    # Actions: 0=HOLD, 1=BUY, 2=SELL
    action_labels = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
    
    # We want to visualize Policy for specific Holding states
    # Let's create a grid for Holding = 0 (Entry Logic) and Holding = 2 (Exit Logic)
    
    trend_labels = ['Down', 'Flat', 'Up']
    rsi_labels = ['Oversold', 'Neutral', 'Overbought']
    
    def get_policy_grid(holding_idx):
        grid = np.zeros((3, 3)) # (Trend, RSI)
        for t_idx in range(3): # Trend
            for r_idx in range(3): # RSI
                state_id = t_idx * 9 + r_idx * 3 + holding_idx
                best_action = np.argmax(q_table[state_id])
                grid[t_idx, r_idx] = best_action
        return grid

    # Create maps
    scenarios = [
        (0, "Entry Policy (When Empty Position)", "3_policy_entry.png"),
        (2, "Exit Policy (When Full Position)", "4_policy_exit.png")
    ]
    
    # Colormap for actions: 0=Hold(Gray), 1=Buy(Green), 2=Sell(Red)
    cmap = mcolors.ListedColormap(['lightgray', 'mediumseagreen', 'salmon'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for holding_idx, title, filename in scenarios:
        grid = get_policy_grid(holding_idx)
        
        plt.figure(figsize=(6, 5))
        
        if HAS_SEABORN:
            ax = sns.heatmap(grid, annot=False, cmap=cmap, norm=norm, 
                             linewidths=1, linecolor='white', cbar=False)
            # Add text annotations
            for y in range(grid.shape[0]):
                for x in range(grid.shape[1]):
                    action_val = int(grid[y, x])
                    text = action_labels[action_val]
                    ax.text(x + 0.5, y + 0.5, text, 
                            ha='center', va='center', color='black', fontweight='bold')
            ax.invert_yaxis()
        else:
            plt.imshow(grid, cmap=cmap, norm=norm, origin='lower')
            # Add text annotations
            for y in range(grid.shape[0]):
                for x in range(grid.shape[1]):
                    action_val = int(grid[y, x])
                    text = action_labels[action_val]
                    plt.text(x, y, text, ha='center', va='center', color='black', fontweight='bold')
            
            # Setup ticks manually for matplotlib
            plt.xticks([0, 1, 2], rsi_labels)
            plt.yticks([0, 1, 2], trend_labels)
            
        if HAS_SEABORN:
            plt.xlabel('RSI Signal')
            plt.ylabel('Price Trend (MA Cross)')
            plt.xticks([0.5, 1.5, 2.5], rsi_labels)
            plt.yticks([0.5, 1.5, 2.5], trend_labels)
        else:
            plt.xlabel('RSI Signal')
            plt.ylabel('Price Trend (MA Cross)')

        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        plt.close()

def plot_state_concept():
    print("Generating State Concept Visualization...")
    # Just a simple graphic listing the dimensions
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    
    text = (
        "State Space Definition (Total 27 States)\n\n"
        "1. Price Trend (3 bins):\n"
        "   - Down (MA5 < MA20)\n"
        "   - Flat (No clear trend)\n"
        "   - Up (MA5 > MA20)\n\n"
        "2. RSI Momentum (3 bins):\n"
        "   - Oversold (< 30)\n"
        "   - Neutral (30 - 70)\n"
        "   - Overbought (> 70)\n\n"
        "3. Current Position (3 bins):\n"
        "   - Empty (0%)\n"
        "   - Partial\n"
        "   - Full (> 80%)"
    )
    
    ax.text(0.1, 0.5, text, fontsize=12, va='center', family='monospace')
    plt.title("Q-Learning State Discretization", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "0_state_discretization.png"), dpi=150)
    plt.close()

def plot_evaluation(test_df, agent):
    print("Generating Evaluation Plots (Equity Curve & Actions)...")
    
    env = TradingEnv(test_df)
    state_vals = env.reset()
    
    # We need to map continuous state to discrete state id for the agent
    def get_state_id(env):
        row = env.df.iloc[env.t]
        _, _, _, state_id = discretize_state(row, env.cash, env.shares, env.initial_cash)
        return state_id

    portfolio_values = [env.initial_cash]
    buy_and_hold_values = [env.initial_cash]
    
    # Track actions for chart
    actions_history = [] # (index, action, price)
    
    # Setup for Buy and Hold
    initial_price = test_df.iloc[0]["Close_original"]
    bnh_shares = env.initial_cash / initial_price
    
    done = False
    while not done:
        state_id = get_state_id(env)
        action = agent.select_action(state_id)
        
        current_price = test_df.iloc[env.t]["Close_original"]
        
        if action == 1: # Buy
            actions_history.append({'t': env.t, 'type': 'BUY', 'price': current_price})
        elif action == 2: # Sell
            actions_history.append({'t': env.t, 'type': 'SELL', 'price': current_price})
            
        _, reward, done, info = env.step(action)
        portfolio_values.append(info['portfolio'])
        
        # Calculate B&H value
        # Note: env.t has incremented by 1 inside step()
        if env.t < len(test_df):
            price_t = test_df.iloc[env.t]["Close_original"]
            buy_and_hold_values.append(bnh_shares * price_t)
        else:
             # Last step
            buy_and_hold_values.append(bnh_shares * test_df.iloc[-1]["Close_original"])

    # 5. Equity Curve
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Q-Learning Agent', color='blue', linewidth=2)
    plt.plot(buy_and_hold_values, label='Buy & Hold (Benchmark)', color='gray', linestyle='--')
    plt.title('Performance Comparison on Test Data')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_equity_curve.png"), dpi=150)
    plt.close()

    # 6. Trading Actions Chart (Zoom in on a busy period or just show all)
    # Let's show the whole test period for context
    plt.figure(figsize=(12, 6))
    
    prices = test_df["Close_original"].values
    plt.plot(prices, label='Stock Price', color='black', alpha=0.5)
    
    buys = [x for x in actions_history if x['type'] == 'BUY']
    sells = [x for x in actions_history if x['type'] == 'SELL']
    
    if buys:
        plt.scatter([x['t'] for x in buys], [x['price'] for x in buys], 
                    marker='^', color='green', s=100, label='Buy Signal', zorder=5)
    if sells:
        plt.scatter([x['t'] for x in sells], [x['price'] for x in sells], 
                    marker='v', color='red', s=100, label='Sell Signal', zorder=5)
        
    plt.title('Agent Trading Decisions on Test Data')
    plt.xlabel('Trading Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_trading_actions.png"), dpi=150)
    plt.close()

def main():
    try:
        plot_state_concept() # Add this call
        df, test_df, agent, q_table = load_data_and_agent()
        
        plot_learning_curve()
        plot_policy_heatmaps(q_table)
        plot_evaluation(test_df, agent)
        
        print(f"\nAll presentation assets generated in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

