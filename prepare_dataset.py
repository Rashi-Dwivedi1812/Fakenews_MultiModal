import pandas as pd

# Load files
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Add labels
fake["label"] = "fake"
real["label"] = "real"

# Keep only required columns
fake = fake[["text", "label"]]
real = real[["text", "label"]]

# Combine
df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)

# Add synthetic metadata (for multi-modal model)
df["followers"] = (1000 * (pd.Series(range(len(df))) % 100 + 1))
df["retweets"] = (df.index % 50)
df["likes"] = (df.index % 200)
df["account_age"] = (df.index % 365)

# Save final dataset
df.to_csv("fake_news_dataset.csv", index=False)

print("Dataset prepared: fake_news_dataset.csv")
