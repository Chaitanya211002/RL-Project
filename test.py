from data_pipeline import load_processed_data

df = load_processed_data()

# Diagnose the issue
print("Index type:", type(df.index))
print("Index is sorted:", df.index.is_monotonic_increasing)
print("Date range:", df.index.min(), "to", df.index.max())
print("First few dates:", df.index[:5])
print("Last few dates:", df.index[-5:])
print("Total rows:", len(df))