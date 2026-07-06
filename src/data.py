"""Data loading. Keep the four Kaggle CSVs in <repo>/data/ (gitignored)."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_raw(data_dir=None):
    """Returns (train, test, features, stores) with parsed dates."""
    d = Path(data_dir) if data_dir else DATA_DIR
    train = pd.read_csv(d / "train.csv", parse_dates=["Date"])
    test = pd.read_csv(d / "test.csv", parse_dates=["Date"])
    features = pd.read_csv(d / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(d / "stores.csv")
    return train, test, features, stores


def make_submission(df, pred_col, path):
    """df needs Store, Dept, Date + pred_col. Writes Kaggle submission csv."""
    sub = df[["Store", "Dept", "Date"]].copy()
    sub["Id"] = (
        sub["Store"].astype(str) + "_" + sub["Dept"].astype(str) + "_"
        + pd.to_datetime(sub["Date"]).dt.strftime("%Y-%m-%d")
    )
    sub["Weekly_Sales"] = df[pred_col].to_numpy()
    sub[["Id", "Weekly_Sales"]].to_csv(path, index=False)
    return path
