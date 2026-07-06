"""Post-processing applied to test predictions (any model)."""
import pandas as pd


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
