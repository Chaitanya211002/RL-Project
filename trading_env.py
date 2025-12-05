import numpy as np


class TradingEnv:
    def __init__(self, df, initial_cash=10000):
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        
        self.reset()

    def reset(self):
        self.t = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.done = False
        self.prev_portfolio = self.initial_cash

        return self._get_state()

    def step(self, action):
        """
        Action Space:
        0 = HOLD
        1 = BUY (use 50% of cash)
        2 = SELL (sell 50% of holdings)
        """

        price = self.df.loc[self.t, "Close_original"]  # ← CHANGE THIS

        # Transaction cost
        cost_rate = 0.001  

        if action == 1:  # BUY
            spend = self.cash * 0.5
            qty = spend / price
            self.shares += qty
            self.cash -= spend * (1 + cost_rate)

        elif action == 2:  # SELL
            sell_qty = self.shares * 0.5
            revenue = sell_qty * price
            self.shares -= sell_qty
            self.cash += revenue * (1 - cost_rate)

        # Advance time
        self.t += 1
        if self.t >= len(self.df) - 1:
            self.done = True

        next_price = self.df.loc[self.t, "Close_original"]  # ← CHANGE THIS
        portfolio_val = self.cash + self.shares * next_price
        reward = (portfolio_val - self.prev_portfolio) / self.prev_portfolio
        self.prev_portfolio = portfolio_val

        return self._get_state(), reward, self.done, {"portfolio": portfolio_val}

    def _get_state(self):
        row = self.df.loc[self.t]
        state = np.array([
            row["Close"],
            row["MA_5"],
            row["MA_20"],
            row["RSI"],
            row["ATR"],
            row["Volume"],
            self.cash / 10000,
            self.shares
        ], dtype=np.float32)
        return state
