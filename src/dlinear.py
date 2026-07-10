"""DLinear (Zeng et al. 2022, "Are Transformers Effective for Time Series
Forecasting?") implemented from the paper — no forecasting library needed.

Architecture: decompose the input window into trend (moving average) and
seasonal remainder, apply one linear layer to each, sum the outputs.
Channel-independent: one shared model, each (Store, Dept) series is forecast
from its own 52-week history in a single shot (direct multi-horizon, same
philosophy as our LightGBM setup — no recursion over the 39-week horizon).

Normalization: per-window standardization (subtract window mean, divide by
window std) — sales scales differ by 100x across series, a shared linear
layer can't handle raw dollars.

Training loss: weighted L1 with weight 1 + 4*IsHoliday on target weeks —
matches the competition WMAE exactly.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.preprocessing import BIG_HOLIDAY_WEEKS


class MovingAvg(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1)

    def forward(self, x):  # (B, L)
        pad_l = (self.kernel - 1) // 2
        pad_r = self.kernel - 1 - pad_l
        front = x[:, :1].repeat(1, pad_l)
        end = x[:, -1:].repeat(1, pad_r)
        return self.avg(torch.cat([front, x, end], dim=1).unsqueeze(1)).squeeze(1)


class DLinear(nn.Module):
    def __init__(self, input_size, horizon, kernel=25):
        super().__init__()
        self.decomp = MovingAvg(kernel)
        self.lin_trend = nn.Linear(input_size, horizon)
        self.lin_season = nn.Linear(input_size, horizon)

    def forward(self, x):  # (B, L) -> (B, H)
        trend = self.decomp(x)
        season = x - trend
        return self.lin_trend(trend) + self.lin_season(season)


def build_wide(df, freq="W-FRI"):
    """Long (Store, Dept, Date, Weekly_Sales) -> wide matrix, one row per
    series, one column per week on a regular weekly grid. Gaps INSIDE a
    series' active range are filled with 0 (dept inactive = no sales);
    weeks before its first / after its last observation stay NaN.
    """
    wide = df.pivot_table(index=["Store", "Dept"], columns="Date",
                          values="Weekly_Sales", aggfunc="sum")
    full = pd.date_range(df["Date"].min(), df["Date"].max(), freq=freq)
    wide = wide.reindex(columns=full)
    arr = wide.to_numpy(dtype=float)
    for i in range(arr.shape[0]):
        valid = np.where(~np.isnan(arr[i]))[0]
        if len(valid):
            lo, hi = valid[0], valid[-1] + 1
            arr[i, lo:hi] = np.nan_to_num(arr[i, lo:hi])
    return pd.DataFrame(arr, index=wide.index, columns=wide.columns)


def _holiday_weights(dates):
    return 1.0 + 4.0 * pd.DatetimeIndex(dates).isin(BIG_HOLIDAY_WEEKS).astype(float)


def make_windows(wide, input_size, horizon, stride=1):
    """All complete sliding windows across all series.

    Returns X (N, input_size), Y (N, horizon), W (N, horizon) holiday weights.
    """
    vals = wide.to_numpy(dtype=float)
    dates = wide.columns
    xs, ys, ws = [], [], []
    n_weeks = vals.shape[1]
    for start in range(0, n_weeks - input_size - horizon + 1, stride):
        xin = vals[:, start:start + input_size]
        yout = vals[:, start + input_size:start + input_size + horizon]
        ok = ~np.isnan(xin).any(axis=1) & ~np.isnan(yout).any(axis=1)
        if not ok.any():
            continue
        xs.append(xin[ok])
        ys.append(yout[ok])
        w = _holiday_weights(dates[start + input_size:start + input_size + horizon])
        ws.append(np.repeat(w[None, :], ok.sum(), axis=0))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ws))


def train_dlinear(wide_train, input_size, horizon, kernel=25, epochs=30,
                  lr=1e-3, batch_size=1024, seed=42, verbose=True):
    """Train on all sliding windows of wide_train. Returns the fitted model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, Y, W = make_windows(wide_train, input_size, horizon)

    # per-window standardization (std floored at $1 so near-constant/zero
    # windows don't blow up the normalized targets)
    mu = X.mean(axis=1, keepdims=True)
    sd = np.maximum(X.std(axis=1, keepdims=True), 1.0)
    Xn = (X - mu) / sd
    Yn = (Y - mu) / sd

    Xt = torch.tensor(Xn, dtype=torch.float32)
    Yt = torch.tensor(Yn, dtype=torch.float32)
    Wt = torch.tensor(W, dtype=torch.float32)
    Sd = torch.tensor(sd, dtype=torch.float32)  # so the loss is in dollars

    model = DLinear(input_size, horizon, kernel)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(Xt[idx])
            loss = (Wt[idx] * (pred - Yt[idx]).abs() * Sd[idx]).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if verbose and (ep + 1) % 10 == 0:
            print(f"  epoch {ep + 1}/{epochs}  weighted-L1 {tot / n:.1f}")
    return model


def forecast(model, wide_train, input_size, horizon, target_dates):
    """Forecast `horizon` weeks after the end of wide_train for every series
    whose last `input_size` weeks are complete. Returns long df
    (Store, Dept, Date, pred); series lacking history are simply absent —
    fall back to seasonal naive for those.
    """
    vals = wide_train.to_numpy(dtype=float)
    xin = vals[:, -input_size:]
    ok = ~np.isnan(xin).any(axis=1)
    mu = xin[ok].mean(axis=1, keepdims=True)
    sd = np.maximum(xin[ok].std(axis=1, keepdims=True), 1.0)
    with torch.no_grad():
        pred = model(torch.tensor((xin[ok] - mu) / sd, dtype=torch.float32)).numpy()
    pred = pred * sd + mu

    idx = wide_train.index[ok]
    out = pd.DataFrame(pred, index=idx, columns=pd.DatetimeIndex(target_dates))
    out = out.stack().rename("pred").reset_index()
    out.columns = ["Store", "Dept", "Date", "pred"]
    return out
