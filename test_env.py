from data_pipeline import load_processed_data, train_val_test_split
from trading_env import TradingEnv

df = load_processed_data()
train, _, _ = train_val_test_split(df)

env = TradingEnv(train)

state = env.reset()
print("Initial state shape:", state.shape)

for i in range(5):
    next_state, reward, done, info = env.step(1)  # BUY action
    print(f"Step {i+1}: reward={reward:.5f}, portfolio={info['portfolio']:.2f}")
