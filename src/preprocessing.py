"""Shared preprocessing for all models.

WalmartPreprocessor is an sklearn transformer that takes RAW rows
(Store, Dept, Date) and produces a model-ready feature frame. All merging
with features.csv / stores.csv, NA handling and lag computation happens
inside — so Pipeline([("prep", ...), ("model", ...)]) can be fit on raw
train rows and run directly on the raw, un-preprocessed test set
(which is what the assignment requires from the registered model).

Design constraints baked in:
- The test horizon is 39 weeks, so every lag feature uses lag >= 39 weeks.
  That makes the model a DIRECT multi-horizon forecaster: no recursion,
  no leakage, one predict() call for the whole test set.
- MarkDown NAs mean "no promotion running" -> fill 0 + presence flag.
- CPI/Unemployment are missing in the test tail -> forward-fill per store.
- LightGBM handles remaining NaNs (e.g. lags of short series) natively.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SUPER_BOWL = pd.to_datetime(["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"])
LABOR_DAY = pd.to_datetime(["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"])
THANKSGIVING = pd.to_datetime(["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"])
CHRISTMAS = pd.to_datetime(["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"])

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]
LAGS = (39, 46, 51, 52, 53)

# all big-4 holiday weeks in one index (the holiday blend uses this too)
BIG_HOLIDAY_WEEKS = SUPER_BOWL.append([LABOR_DAY, THANKSGIVING, CHRISTMAS])

# Holiday-aligned lags: the big holidays drift across week-of-year numbers,
# so the plain 52-week lag lands one week off exactly on the 5x-weighted
# weeks. Align by holiday instead: "this week is Thanksgiving+1 of 2012 ->
# look at Thanksgiving+1 of 2011". Offset ranges are capped so the
# Thanksgiving (max +1) and Christmas (min -3) windows never overlap.
_HOLIDAY_OFFSETS = {
    "superbowl": (SUPER_BOWL, range(-2, 2)),
    "laborday": (LABOR_DAY, range(-2, 2)),
    "thanksgiving": (THANKSGIVING, range(-2, 2)),
    "christmas": (CHRISTMAS, range(-3, 2)),
}


def _build_holiday_maps():
    align, relpos = {}, {}
    for dates, offsets in _HOLIDAY_OFFSETS.values():
        dates = list(dates)
        for off in offsets:
            delta = pd.Timedelta(days=7 * off)
            for cur in dates:
                relpos[cur + delta] = off
            for prev, cur in zip(dates[:-1], dates[1:]):
                align[cur + delta] = prev + delta
    return align, relpos


_HOLIDAY_ALIGN, _HOLIDAY_RELPOS = _build_holiday_maps()

STORE_CATS = list(range(1, 46))
DEPT_CATS = list(range(1, 100))
TYPE_CATS = ["A", "B", "C"]


def make_xyw(df):
    """Split a raw train/val frame into (X, y, sample_weight).

    sample_weight = 1 + 4*IsHoliday so training optimizes the same thing
    the competition metric measures.
    """
    X = df[["Store", "Dept", "Date"]].copy()
    y = df["Weekly_Sales"].to_numpy(dtype=float)
    w = 1.0 + 4.0 * df["IsHoliday"].to_numpy(dtype=float)
    return X, y, w


class WalmartPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, features_df=None, stores_df=None, holiday_lags=False):
        # features_df/stores_df kept as constructor params so they get pickled
        # with the pipeline: the registered model is fully self-contained.
        # holiday_lags: aligned-by-holiday lag features. Fold-1 ablation showed
        # they HURT (holiday MAE 2355 -> 2464; one aligned year is too noisy,
        # the model over-trusts it), so default off; kept for the ablation story.
        self.features_df = features_df
        self.stores_df = stores_df
        self.holiday_lags = holiday_lags

    def fit(self, X, y):
        hist = X[["Store", "Dept", "Date"]].copy()
        hist["Date"] = pd.to_datetime(hist["Date"])
        hist["Weekly_Sales"] = np.asarray(y, dtype=float)
        self.history_ = hist

        g = hist.groupby(["Store", "Dept"])["Weekly_Sales"]
        self.series_stats_ = g.agg(
            series_mean="mean", series_median="median", series_std="std"
        ).reset_index()

        woy = hist["Date"].dt.isocalendar().week.astype(int)
        self.dept_woy_mean_ = (
            hist.assign(woy=woy)
            .groupby(["Dept", "woy"])["Weekly_Sales"]
            .mean()
            .rename("dept_woy_mean")
            .reset_index()
        )
        self.store_mean_ = (
            hist.groupby("Store")["Weekly_Sales"].mean().rename("store_mean").reset_index()
        )

        f = self.features_df.copy()
        f["Date"] = pd.to_datetime(f["Date"])
        f = f.sort_values(["Store", "Date"])
        f[["CPI", "Unemployment"]] = f.groupby("Store")[["CPI", "Unemployment"]].ffill()
        for c in MARKDOWN_COLS:
            f[c + "_present"] = f[c].notna().astype(int)
            f[c] = f[c].fillna(0.0)
        self.features_clean_ = f
        return self

    def transform(self, X):
        out = X[["Store", "Dept", "Date"]].copy()
        out["Date"] = pd.to_datetime(out["Date"])

        out = out.merge(self.stores_df, on="Store", how="left")
        out = out.merge(self.features_clean_, on=["Store", "Date"], how="left")

        d = out["Date"]
        out["year"] = d.dt.year
        out["month"] = d.dt.month
        out["weekofyear"] = d.dt.isocalendar().week.astype(int)
        out["is_superbowl"] = d.isin(SUPER_BOWL).astype(int)
        out["is_laborday"] = d.isin(LABOR_DAY).astype(int)
        out["is_thanksgiving"] = d.isin(THANKSGIVING).astype(int)
        out["is_christmas"] = d.isin(CHRISTMAS).astype(int)
        # the big sales spikes happen the week BEFORE Thanksgiving/Christmas
        out["is_pre_thanksgiving"] = d.isin(THANKSGIVING - pd.Timedelta(days=7)).astype(int)
        out["is_pre_christmas"] = d.isin(CHRISTMAS - pd.Timedelta(days=7)).astype(int)
        xmas = pd.to_datetime(d.dt.year.astype(str) + "-12-25")
        days = (xmas - d).dt.days
        out["days_to_christmas"] = np.where(days < 0, days + 365, days)

        # lag features from the stored history (all lags >= 39 -> direct forecasting)
        for L in LAGS:
            lagged = self.history_.copy()
            lagged["Date"] = lagged["Date"] + pd.Timedelta(days=7 * L)
            lagged = lagged.rename(columns={"Weekly_Sales": f"lag_{L}"})
            out = out.merge(lagged, on=["Store", "Dept", "Date"], how="left")
        out["yearly_smooth"] = out[["lag_51", "lag_52", "lag_53"]].mean(axis=1)

        if self.holiday_lags:
            # holiday-aligned lag: same series, same position relative to the
            # PREVIOUS year's holiday (~52-53 weeks back, so still leak-free)
            out["hol_relpos"] = d.map(_HOLIDAY_RELPOS).fillna(9).astype(int)
            out["_hol_aligned_date"] = d.map(_HOLIDAY_ALIGN)
            aligned = self.history_.rename(
                columns={"Date": "_hol_aligned_date", "Weekly_Sales": "hol_lag_1y"}
            )
            out = out.merge(
                aligned, on=["Store", "Dept", "_hol_aligned_date"], how="left"
            ).drop(columns="_hol_aligned_date")

        out = out.merge(self.series_stats_, on=["Store", "Dept"], how="left")
        out = out.merge(
            self.dept_woy_mean_,
            left_on=["Dept", "weekofyear"],
            right_on=["Dept", "woy"],
            how="left",
        ).drop(columns="woy")
        out = out.merge(self.store_mean_, on="Store", how="left")

        out["Store"] = pd.Categorical(out["Store"], categories=STORE_CATS)
        out["Dept"] = pd.Categorical(out["Dept"], categories=DEPT_CATS)
        out["Type"] = pd.Categorical(out["Type"], categories=TYPE_CATS)
        out["IsHoliday"] = out["IsHoliday"].astype(float)

        return out.drop(columns=["Date"])
