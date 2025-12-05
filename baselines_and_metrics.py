import numpy as np

def buy_and_hold(df, initial_balance=10000):
    start = df["Close"].iloc[0]
    end = df["Close"].iloc[-1]
    shares = initial_balance / start
    final_value = shares * end
    roi = (final_value - initial_balance) / initial_balance
    return final_value, roi


def random_strategy(df, initial_balance=10000):
    portfolio = initial_balance
    shares = 0

    for price in df["Close"]:
        a = np.random.choice([0, 1, 2])  # 0 hold, 1 buy, 2 sell

        if a == 1:  # buy 50%
            qty = (portfolio * 0.5) / price
            shares += qty
            portfolio -= qty * price

        elif a == 2:  # sell 50%
            qty = shares * 0.5
            shares -= qty
            portfolio += qty * price

    # Include value of remaining shares
    final_value = portfolio + shares * df["Close"].iloc[-1]
    roi = (final_value - initial_balance) / initial_balance
    return final_value, roi


def total_return(values):
    return (values[-1] - values[0]) / values[0]


def sharpe_ratio(values):
    daily_returns = np.diff(values) / values[:-1]
    return (daily_returns.mean() / (daily_returns.std() + 1e-8)) * np.sqrt(252)


def max_drawdown(values):
    peak = values[0]
    max_dd = 0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)
    return max_dd


if __name__ == "__main__":
    from data_pipeline import load_processed_data, train_val_test_split

    df = load_processed_data()
    train, val, test = train_val_test_split(df)

    print("\n=== Baseline Strategy Results (on Test Set) ===")

    bh_value, bh_roi = buy_and_hold(test)
    print(f"Buy & Hold → Final Value = ${bh_value:.2f}, ROI = {bh_roi:.2%}")

    rand_value, rand_roi = random_strategy(test)
    print(f"Random Strategy → Final Value = ${rand_value:.2f}, ROI = {rand_roi:.2%}")

    print("\nMetrics:")
    values = list(test["Close"])
    print(f"Total Return = {total_return(values):.2%}")
    print(f"Sharpe Ratio = {sharpe_ratio(values):.4f}")
    print(f"Max Drawdown = {max_drawdown(values):.2%}")
