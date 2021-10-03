import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("./data/extracted_sentence.csv")
    data_sample = data.sample(n=5000, random_state=42)
    data_sample.to_csv("./data/test_medium.csv", index=False)