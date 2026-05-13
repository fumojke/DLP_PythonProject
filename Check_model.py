import pandas as pd

df = pd.read_csv("dataset_old.csv")
print(df["label"].value_counts())
