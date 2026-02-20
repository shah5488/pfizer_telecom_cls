import pandas as pd
from src.config import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH)
    df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    return df
