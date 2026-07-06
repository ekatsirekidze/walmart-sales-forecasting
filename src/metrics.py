"""Competition metric. EVERYONE uses this — no other metric for model comparison."""
import numpy as np


def wmae(y_true, y_pred, is_holiday):
    """Weighted MAE: holiday weeks weigh 5x, exactly as Kaggle scores it."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.where(np.asarray(is_holiday, dtype=bool), 5.0, 1.0)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))


def wmae_report(y_true, y_pred, is_holiday):
    """WMAE plus the holiday / non-holiday MAE split — log all three to MLflow."""
    is_holiday = np.asarray(is_holiday, dtype=bool)
    ae = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    return {
        "wmae": wmae(y_true, y_pred, is_holiday),
        "mae_holiday": float(ae[is_holiday].mean()) if is_holiday.any() else float("nan"),
        "mae_nonholiday": float(ae[~is_holiday].mean()),
    }
