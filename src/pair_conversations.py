import pandas as pd
from config import RAW_DATA_PATH

df = pd.read_csv(RAW_DATA_PATH)

inbound = df[df["inbound"] == True]
apple_customer = inbound[inbound["text"].str.contains("@AppleSupport", na=False)]

apple_replies = df[(df["inbound"] == False) & (df["author_id"] == "AppleSupport")]

pairs = apple_customer.merge(
  apple_replies,
  left_on="tweet_id",
  right_on="in_response_to_tweet_id",
  suffixes=("_customer", "_reply")
)

print(f"{len(pairs)} matched customer->AppleSupport reply pairs")
print(pairs[["text_customer", "text_reply"]].sample(10, random_state=1).to_string())

#save it - for intent labeling, retrieval, and golden set sampling set later
pairs.to_csv("data/processed/apple_pairs.csv", index=False)