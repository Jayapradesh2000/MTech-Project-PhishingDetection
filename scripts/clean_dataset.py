import pandas as pd

# Load your dataset
df = pd.read_csv("data/dataset_urls.csv")

print("Before cleaning:", len(df), "rows")

# Drop duplicate URLs
df = df.drop_duplicates(subset=['url']).reset_index(drop=True)

print("After removing duplicates:", len(df), "rows")

# Save as a new cleaned file
df.to_csv("data/dataset_urls_clean.csv", index=False)

print("✅ Saved cleaned dataset to data/dataset_urls_clean.csv")
