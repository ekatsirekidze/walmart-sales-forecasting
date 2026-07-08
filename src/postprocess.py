"""Post-processing applied to test predictions (any model)."""
import numpy as np
import pandas as pd

from src.preprocessing import BIG_HOLIDAY_WEEKS


def naive_lag52(history, target):
    """Seasonal-naive prediction (lag 52, fallbacks 53/51) for target rows.

    Returns an array aligned to target; NaN where no history exists.
    """
    hist = history[["Store", "Dept", "Date", "Weekly_Sales"]]
    out = target[["Store", "Dept", "Date"]].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    for L in (52, 53, 51):
        lagged = hist.copy()
        lagged["Date"] = lagged["Date"] + pd.Timedelta(days=7 * L)
        out = out.merge(
            lagged.rename(columns={"Weekly_Sales": f"_l{L}"}),
            on=["Store", "Dept", "Date"],
            how="left",
        )
    return out["_l52"].fillna(out["_l53"]).fillna(out["_l51"]).to_numpy()


def blend_holiday_naive(df, history, pred_col="pred", weight=0.5, holiday_dates=None):
    """On big-holiday weeks, blend the model prediction with seasonal-naive.

    Motivated by the Fold-1 finding that naive lag-52 beats the trained model
    on holiday weeks (the 5x-weighted ones). weight=0 -> pure model,
    weight=1 -> pure naive. Rows without naive history are left untouched.
    """
    out = df.copy()
    d = pd.to_datetime(out["Date"])
    if holiday_dates is None:
        holiday_dates = BIG_HOLIDAY_WEEKS
    naive = naive_lag52(history, out)
    mask = d.isin(holiday_dates).to_numpy() & ~np.isnan(naive)
    out.loc[mask, pred_col] = (
        (1.0 - weight) * out.loc[mask, pred_col].to_numpy() + weight * naive[mask]
    )
    return out


def apply_christmas_shift(df, pred_col="pred", shift_days=2.5):
    """The famous 1st-place calendar fix.

    In the training years the week ending right after Christmas contained
    0 (2010) or 1 (2011) pre-Christmas shopping days; in the test year the
    week ending 2012-12-28 contains 3 (Dec 22-24). Any model trained on
    history therefore under-predicts that 5x-weighted week. Fix: move part
    of the huge week-51 forecast into week 52.

        adjusted_52 = pred_52 + (shift_days/7) * (pred_51 - pred_52)

    applied per (Store, Dept), only when pred_51 > pred_52.
    """
    out = df.copy()
    d = pd.to_datetime(out["Date"])
    w51 = out.loc[d == "2012-12-21", ["Store", "Dept", pred_col]].rename(
        columns={pred_col: "_p51"}
    )
    out = out.merge(w51, on=["Store", "Dept"], how="left")
    mask = (d == "2012-12-28") & (out["_p51"] > out[pred_col])
    out.loc[mask, pred_col] = (
        out.loc[mask, pred_col]
        + (shift_days / 7.0) * (out.loc[mask, "_p51"] - out.loc[mask, pred_col])
    )
    return out.drop(columns="_p51")
