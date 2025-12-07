import numpy as np


def price_trend_bin(row, delta: float = 0.005) -> int:
    """
    0 = Down, 1 = Flat, 2 = Up based on short/long moving average spread.
    delta is the relative threshold on (MA5 - MA20) / |MA20|.
    """
    short_ma = row.get("MA_5_original", np.nan)
    long_ma = row.get("MA_20_original", np.nan)

    if np.isnan(short_ma) or np.isnan(long_ma):
        return 1  # fallback to Flat

    spread = (short_ma - long_ma) / (abs(long_ma) + 1e-8)
    if spread > delta:
        return 2  # Up
    if spread < -delta:
        return 0  # Down
    return 1  # Flat


def rsi_bin(row) -> int:
    """
    0 = Oversold (<30), 1 = Neutral (30-70), 2 = Overbought (>70).
    """
    rsi = row.get("RSI_original", row.get("RSI", np.nan))
    if np.isnan(rsi):
        return 1  # Neutral fallback
    if rsi < 30:
        return 0
    if rsi > 70:
        return 2
    return 1


def holding_bin(cash: float, shares: float, price: float, full_threshold: float = 0.8) -> int:
    """
    0 = Flat, 1 = Partial, 2 = Full based on equity allocation.
    full_threshold: ratio of equity value / total portfolio to be considered 'Full'.
    """
    if price <= 0:
        return 0

    equity_value = shares * price
    portfolio_value = cash + equity_value
    if portfolio_value <= 1e-8 or shares <= 1e-6:
        return 0

    equity_ratio = equity_value / portfolio_value
    if equity_ratio >= full_threshold:
        return 2
    return 1


def discretize_state(row, cash: float, shares: float, initial_cash: float) -> tuple[int, int, int, int]:
    """
    Convert continuous info into discrete bins and a compact state_id.
    Returns (price_bin, rsi_bin, holding_bin, state_id in [0,26]).
    """
    price_bin_val = price_trend_bin(row)
    rsi_bin_val = rsi_bin(row)
    price = row.get("Close_original", row.get("Close", 0.0))
    holding_bin_val = holding_bin(cash, shares, price)

    state_id = price_bin_val * 9 + rsi_bin_val * 3 + holding_bin_val
    return price_bin_val, rsi_bin_val, holding_bin_val, int(state_id)

