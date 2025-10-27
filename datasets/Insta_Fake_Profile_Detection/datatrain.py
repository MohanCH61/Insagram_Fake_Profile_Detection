import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

np.random.seed(42)
n = 1000

# Create dummy dataset with all 11 columns
data = {
    "profile_pic": np.random.randint(0, 2, n),
    "nums_length_username": np.random.randint(5, 15, n),
    "fullname_words": np.random.randint(1, 4, n),
    "nums_length_fullname": np.random.randint(5, 20, n),
    "name_equals_username": np.random.randint(0, 2, n),
    "description_length": np.random.randint(0, 300, n),
    "external_url": np.random.randint(0, 2, n),
    "private": np.random.randint(0, 2, n),
    "posts": np.random.randint(0, 500, n),
    "followers": np.random.randint(0, 5000, n),
    "follows": np.random.randint(0, 5000, n)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# ✅ Normalize the two specific columns (0–1 float range)
df["nums_length_username"] = df["nums_length_username"] / df["nums_length_username"].max()
df["nums_length_fullname"] = df["nums_length_fullname"] / df["nums_length_fullname"].max()

# Label (simple rule)
df["fake"] = ((df["followers"] < 100) & (df["follows"] > 1000)).astype(int)

# Save to files
os.makedirs("datasets/Insta_Fake_Profile_Detection", exist_ok=True)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv("datasets/Insta_Fake_Profile_Detection/train.csv", index=False)
test_df.to_csv("datasets/Insta_Fake_Profile_Detection/test.csv", index=False)

print("✅ train.csv and test.csv created successfully with normalized values!")
