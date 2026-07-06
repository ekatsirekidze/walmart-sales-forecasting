"""Shared validation folds. EVERYONE reports Fold 1 (primary) + Fold 2 (secondary).

Both folds are 39-week horizons — the exact shape of the Kaggle test set
(2012-11-02 .. 2013-07-26). Fold 1 contains Thanksgiving, Christmas and the
Super Bowl, like the test set does; decisions are made on Fold 1.
Fold 2 has no big holidays and serves as a regular-weeks sanity check.
"""
import pandas as pd

FOLDS = {
    1: {"train_end": "2011-10-28", "val_start": "2011-11-04", "val_end": "2012-07-27"},
    2: {"train_end": "2012-01-27", "val_start": "2012-02-03", "val_end": "2012-10-26"},
}


def split_fold(df, fold, date_col="Date"):
    """Expanding-window split: returns (train_df, val_df) for the given fold."""
    f = FOLDS[fold]
    d = pd.to_datetime(df[date_col])
    train = df[d <= f["train_end"]].copy()
    val = df[(d >= f["val_start"]) & (d <= f["val_end"])].copy()
    return train, val
