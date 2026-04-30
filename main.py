from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# Main competition target. COGS is only filled to keep submission format valid.
TARGET = "Revenue"

# Long horizon forecast: avoid very short lags to reduce recursive error propagation.
LAGS = [28, 56, 91, 182, 364, 365, 366, 730]
ROLL_WINDOWS = [28, 56, 91, 182, 365]


@dataclass
class LoadedData:
    sales: pd.DataFrame
    sample: pd.DataFrame
    future_dates_chrono: pd.Series
    sample_original_order: pd.Series
    base_date: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Datathon forecasting pipeline: baseline + recursive YoY LightGBM + direct seasonal-anchor LightGBM."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Folder containing sales.csv and sample_submission.csv. Default: data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Folder to save submissions, plots, and feature importances. Default: outputs",
    )
    parser.add_argument(
        "--run-cv",
        action="store_true",
        help="Run rolling time-series validation. Slower but useful for the report.",
    )
    parser.add_argument(
        "--run-shap",
        action="store_true",
        help="Create SHAP summary plot if shap is installed. Optional.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=1400,
        help="Number of LightGBM trees. Default: 1400",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.018,
        help="LightGBM learning rate. Default: 0.018",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for LightGBM. Default: -1",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        default="yoy_ratio",
        choices=["direct", "yoy_ratio"],
        help=(
            "Target transformation for recursive model. "
            "direct learns log1p(Revenue). "
            "yoy_ratio learns log1p(Revenue) - log1p(Revenue_lag_365). "
            "Default: yoy_ratio"
        ),
    )
    parser.add_argument(
        "--recursive-weight",
        type=float,
        default=0.20,
        help="Weight for recursive YoY LightGBM in final ensemble. Default: 0.20",
    )
    parser.add_argument(
        "--anchor-weight",
        type=float,
        default=0.45,
        help="Weight for direct seasonal-anchor LightGBM in final ensemble. Default: 0.45",
    )
    parser.add_argument(
        "--baseline-weight",
        type=float,
        default=0.35,
        help="Weight for seasonal naive baseline in final ensemble. Default: 0.35",
    )
    return parser.parse_args()


def import_lightgbm():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor
    except ImportError as exc:
        raise ImportError(
            "LightGBM is not installed. Run:\n"
            "python -m pip install lightgbm"
        ) from exc


def normalize_weights(recursive_weight: float, anchor_weight: float, baseline_weight: float) -> Tuple[float, float, float]:
    weights = np.asarray([recursive_weight, anchor_weight, baseline_weight], dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()

    if total <= 0:
        return 0.20, 0.45, 0.35

    weights = weights / total
    return float(weights[0]), float(weights[1]), float(weights[2])


def ensure_required_files(data_dir: Path) -> None:
    required = ["sales.csv", "sample_submission.csv"]
    missing = [name for name in required if not (data_dir / name).exists()]

    if missing:
        msg = [
            f"Missing required file(s) in {data_dir.resolve()}: {missing}",
            "Expected project structure:",
            "  project_folder/",
            "    main.py",
            "    requirements.txt",
            "    data/",
            "      sales.csv",
            "      sample_submission.csv",
        ]
        raise FileNotFoundError("\n".join(msg))


def load_and_validate(data_dir: Path) -> LoadedData:
    ensure_required_files(data_dir)

    sales = pd.read_csv(data_dir / "sales.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    sales["Date"] = pd.to_datetime(sales["Date"])
    sample["Date"] = pd.to_datetime(sample["Date"])

    sales = sales.sort_values("Date").reset_index(drop=True)
    sample = sample.reset_index(drop=True)  # keep sample order

    expected_cols = ["Date", "Revenue", "COGS"]

    if list(sales.columns) != expected_cols:
        raise ValueError(f"sales.csv columns must be {expected_cols}, got {sales.columns.tolist()}")

    if list(sample.columns) != expected_cols:
        raise ValueError(
            f"sample_submission.csv columns must be {expected_cols}, got {sample.columns.tolist()}"
        )

    if not sales["Date"].is_unique:
        raise ValueError("sales.csv has duplicated dates.")
    if not sample["Date"].is_unique:
        raise ValueError("sample_submission.csv has duplicated dates.")

    sales_full_dates = pd.date_range(sales["Date"].min(), sales["Date"].max(), freq="D")
    sample_full_dates = pd.date_range(sample["Date"].min(), sample["Date"].max(), freq="D")

    if len(sales_full_dates) != len(sales):
        raise ValueError("sales.csv is not a complete daily series.")
    if len(sample_full_dates) != len(sample):
        raise ValueError("sample_submission.csv is not a complete daily series.")

    print("Data loaded")
    print(f"  sales shape: {sales.shape}")
    print(f"  sample shape: {sample.shape}")
    print(f"  sales date range: {sales['Date'].min().date()} -> {sales['Date'].max().date()}")
    print(f"  test date range: {sample['Date'].min().date()} -> {sample['Date'].max().date()}")

    return LoadedData(
        sales=sales,
        sample=sample,
        future_dates_chrono=pd.Series(sample_full_dates),
        sample_original_order=sample["Date"].copy(),
        base_date=sales["Date"].min(),
    )


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
    r2 = r2_score(y_true_arr, y_pred_arr)

    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def compute_yoy_factor(
    history_df: pd.DataFrame,
    target: str,
    lookback_days: int = 90,
    clip: Tuple[float, float] = (0.70, 1.30),
) -> float:
    s = history_df.set_index("Date")[target].sort_index().astype(float)

    end = s.index.max()
    recent_start = end - pd.Timedelta(days=lookback_days - 1)
    previous_start = recent_start - pd.Timedelta(days=365)
    previous_end = end - pd.Timedelta(days=365)

    recent_mean = s.loc[recent_start:end].mean()
    previous_mean = s.loc[previous_start:previous_end].mean()

    if pd.isna(recent_mean) or pd.isna(previous_mean) or previous_mean <= 0:
        return 1.0

    factor = recent_mean / previous_mean
    return float(np.clip(factor, clip[0], clip[1]))


def compute_yoy_factor_asof(
    hist: pd.Series,
    asof_date: pd.Timestamp,
    lookback_days: int = 180,
    clip: Tuple[float, float] = (0.80, 1.20),
) -> float:
    """
    Growth factor available as of a date.
    For training rows, this avoids using future rows relative to the row.
    For future rows, it uses the last available training date.
    """
    if len(hist) == 0:
        return 1.0

    asof_date = pd.Timestamp(asof_date)
    end = min(asof_date - pd.Timedelta(days=1), hist.index.max())

    recent_start = end - pd.Timedelta(days=lookback_days - 1)
    previous_start = recent_start - pd.Timedelta(days=365)
    previous_end = end - pd.Timedelta(days=365)

    recent = hist.loc[recent_start:end]
    previous = hist.loc[previous_start:previous_end]

    if len(recent) < 30 or len(previous) < 30:
        return 1.0

    recent_mean = recent.mean()
    previous_mean = previous.mean()

    if pd.isna(recent_mean) or pd.isna(previous_mean) or previous_mean <= 0:
        return 1.0

    return float(np.clip(recent_mean / previous_mean, clip[0], clip[1]))


def seasonal_naive_recursive(
    history_df: pd.DataFrame,
    future_dates: Iterable[pd.Timestamp],
    target: str,
    seasonal_lag: int = 365,
    use_trend: bool = True,
) -> np.ndarray:
    hist = history_df[["Date", target]].copy()
    hist = hist.sort_values("Date").set_index("Date")[target].astype(float)

    factor = compute_yoy_factor(history_df, target) if use_trend else 1.0
    preds = []

    for d in pd.to_datetime(future_dates):
        d = pd.Timestamp(d)
        ref_date = d - pd.Timedelta(days=seasonal_lag)

        if ref_date in hist.index:
            pred = hist.loc[ref_date] * factor
        else:
            pred = hist.tail(28).mean()

        pred = max(float(pred), 0.0)
        preds.append(pred)
        hist.loc[d] = pred

    return np.asarray(preds, dtype=float)


def validate_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    sample_original_order: pd.Series,
) -> None:
    if submission.shape != sample.shape:
        raise ValueError(f"Submission shape mismatch: {submission.shape} vs {sample.shape}")
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample_submission.csv")
    if not submission["Date"].equals(sample_original_order):
        raise ValueError("Date order was changed. Keep sample_submission.csv order.")

    for col in ["Revenue", "COGS"]:
        if submission[col].isna().any():
            raise ValueError(f"Submission has NaN in {col}")
        if (submission[col] < 0).any():
            raise ValueError(f"Submission has negative values in {col}")


def save_baseline_submission(data: LoadedData, output_dir: Path) -> Path:
    print("Creating seasonal naive baseline submission...")

    base_pred_df = pd.DataFrame({"Date": data.future_dates_chrono})
    base_pred_df["Revenue"] = seasonal_naive_recursive(
        history_df=data.sales,
        future_dates=data.future_dates_chrono,
        target="Revenue",
        use_trend=True,
    )
    base_pred_df["COGS"] = seasonal_naive_recursive(
        history_df=data.sales,
        future_dates=data.future_dates_chrono,
        target="COGS",
        use_trend=True,
    )

    pred_map = base_pred_df.set_index("Date")
    baseline_submission = data.sample.copy()
    baseline_submission["Revenue"] = baseline_submission["Date"].map(pred_map["Revenue"])
    baseline_submission["COGS"] = baseline_submission["Date"].map(pred_map["COGS"])
    baseline_submission["Revenue"] = baseline_submission["Revenue"].clip(lower=0).round(2)
    baseline_submission["COGS"] = baseline_submission["COGS"].clip(lower=0).round(2)

    validate_submission(baseline_submission, data.sample, data.sample_original_order)

    path = output_dir / "submission_baseline.csv"
    baseline_submission.to_csv(path, index=False)
    print(f"Saved baseline: {path}")
    return path


def make_calendar_features(dates: Iterable[pd.Timestamp], base_date: pd.Timestamp) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    X = pd.DataFrame(index=dates.index)

    X["time_idx"] = (dates - base_date).dt.days
    X["year"] = dates.dt.year
    X["month"] = dates.dt.month
    X["quarter"] = dates.dt.quarter
    X["weekofyear"] = dates.dt.isocalendar().week.astype(int)
    X["dayofyear"] = dates.dt.dayofyear
    X["dayofmonth"] = dates.dt.day
    X["dayofweek"] = dates.dt.dayofweek
    X["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    X["is_month_start"] = dates.dt.is_month_start.astype(int)
    X["is_month_end"] = dates.dt.is_month_end.astype(int)

    # Stronger annual seasonality basis.
    for k in [1, 2, 3, 4, 5, 6]:
        X[f"sin_doy_{k}"] = np.sin(2 * np.pi * k * X["dayofyear"] / 365.25)
        X[f"cos_doy_{k}"] = np.cos(2 * np.pi * k * X["dayofyear"] / 365.25)

    X["sin_dow"] = np.sin(2 * np.pi * X["dayofweek"] / 7)
    X["cos_dow"] = np.cos(2 * np.pi * X["dayofweek"] / 7)

    return X


def build_feature_frame(df: pd.DataFrame, target: str, base_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values("Date").reset_index(drop=True).copy()
    X = make_calendar_features(df["Date"], base_date=base_date)
    s = df[target].astype(float)

    for lag in LAGS:
        X[f"{target}_lag_{lag}"] = s.shift(lag)

    shifted = s.shift(1)
    for w in ROLL_WINDOWS:
        X[f"{target}_roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        X[f"{target}_roll_std_{w}"] = shifted.rolling(w, min_periods=2).std()
        X[f"{target}_roll_min_{w}"] = shifted.rolling(w, min_periods=1).min()
        X[f"{target}_roll_max_{w}"] = shifted.rolling(w, min_periods=1).max()
        X[f"{target}_roll_median_{w}"] = shifted.rolling(w, min_periods=1).median()

    X[f"{target}_lag365_to_lag730"] = X[f"{target}_lag_365"] / X[f"{target}_lag_730"].replace(0, np.nan)
    X[f"{target}_lag364_to_lag365"] = X[f"{target}_lag_364"] / X[f"{target}_lag_365"].replace(0, np.nan)

    y = df[target].astype(float)
    return X, y


def make_one_step_features(
    history_series: pd.Series,
    date: pd.Timestamp,
    target: str,
    feature_cols: List[str],
    base_date: pd.Timestamp,
) -> pd.DataFrame:
    date = pd.Timestamp(date)
    X = make_calendar_features(pd.Series([date]), base_date=base_date)

    for lag in LAGS:
        ref_date = date - pd.Timedelta(days=lag)
        X[f"{target}_lag_{lag}"] = history_series.get(ref_date, np.nan)

    past = history_series.loc[: date - pd.Timedelta(days=1)]
    for w in ROLL_WINDOWS:
        vals = past.tail(w)
        X[f"{target}_roll_mean_{w}"] = vals.mean() if len(vals) > 0 else np.nan
        X[f"{target}_roll_std_{w}"] = vals.std() if len(vals) > 1 else np.nan
        X[f"{target}_roll_min_{w}"] = vals.min() if len(vals) > 0 else np.nan
        X[f"{target}_roll_max_{w}"] = vals.max() if len(vals) > 0 else np.nan
        X[f"{target}_roll_median_{w}"] = vals.median() if len(vals) > 0 else np.nan

    X[f"{target}_lag365_to_lag730"] = X[f"{target}_lag_365"] / X[f"{target}_lag_730"].replace(0, np.nan)
    X[f"{target}_lag364_to_lag365"] = X[f"{target}_lag_364"] / X[f"{target}_lag_365"].replace(0, np.nan)

    return X.reindex(columns=feature_cols)


def make_anchor_features(
    dates: Iterable[pd.Timestamp],
    history_df: pd.DataFrame,
    target: str,
    base_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Direct seasonal-anchor features.
    This model does NOT recursively depend on previous forecast days.
    Each future date is anchored to previous-year / previous-2-year / previous-3-year same-period values.
    """
    hist = (
        history_df[["Date", target]]
        .sort_values("Date")
        .set_index("Date")[target]
        .astype(float)
    )

    dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    X = make_calendar_features(dates, base_date=base_date)

    anchors = []
    anchor_lag_used = []
    anchor_growth_used = []
    anchor_candidate_count = []

    for d in dates:
        d = pd.Timestamp(d)
        growth = compute_yoy_factor_asof(hist, d, lookback_days=180, clip=(0.80, 1.20))

        candidates = []

        # Around same date in previous 1, 2, and 3 years.
        # Weights favor last-year seasonality, but allow older history when last year is unavailable.
        for lag, years_back, weight in [
            (364, 1, 0.55),
            (365, 1, 0.75),
            (366, 1, 0.55),
            (729, 2, 0.25),
            (730, 2, 0.35),
            (731, 2, 0.25),
            (1094, 3, 0.08),
            (1095, 3, 0.12),
            (1096, 3, 0.08),
        ]:
            ref_date = d - pd.Timedelta(days=lag)

            if ref_date in hist.index:
                ref_value = hist.loc[ref_date]
                adjusted_value = ref_value * (growth ** years_back)
                candidates.append((weight, adjusted_value, lag))

        if candidates:
            total_weight = sum(w for w, _, _ in candidates)
            anchor = sum(w * v for w, v, _ in candidates) / total_weight
            avg_lag = sum(w * lag for w, _, lag in candidates) / total_weight
            n_candidates = len(candidates)
        else:
            anchor = hist.tail(365).median()
            avg_lag = 999
            n_candidates = 0

        anchors.append(anchor)
        anchor_lag_used.append(avg_lag)
        anchor_growth_used.append(growth)
        anchor_candidate_count.append(n_candidates)

    X["anchor_revenue"] = anchors
    X["anchor_log"] = np.log1p(X["anchor_revenue"])
    X["anchor_lag_used"] = anchor_lag_used
    X["anchor_growth_used"] = anchor_growth_used
    X["anchor_candidate_count"] = anchor_candidate_count
    X["anchor_missing_flag"] = (X["anchor_candidate_count"] == 0).astype(int)

    return X


def get_lgbm_params(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "objective": "regression",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": 15,
        "min_child_samples": 50,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.5,
        "reg_lambda": 10.0,
        "random_state": SEED,
        "n_jobs": args.n_jobs,
        "verbose": -1,
    }


def train_recursive_lgbm(
    train_df: pd.DataFrame,
    target: str,
    base_date: pd.Timestamp,
    args: argparse.Namespace,
    min_history_days: int = 730,
):
    LGBMRegressor = import_lightgbm()

    train_df = train_df.sort_values("Date").reset_index(drop=True).copy()
    X, y = build_feature_frame(train_df, target=target, base_date=base_date)

    first_train_date = train_df["Date"].min() + pd.Timedelta(days=min_history_days)
    mask = train_df["Date"] >= first_train_date
    y_log = np.log1p(y)

    if args.target_mode == "yoy_ratio":
        lag_col = f"{target}_lag_365"
        lag_365 = X[lag_col].astype(float)
        valid_lag_mask = lag_365.notna() & (lag_365 > 0)
        mask = mask & valid_lag_mask
        y_train = y_log.loc[mask] - np.log1p(lag_365.loc[mask])
    else:
        y_train = y_log.loc[mask]

    X_train = X.loc[mask].copy()
    y_train = y_train.copy()

    if len(X_train) == 0:
        raise ValueError(f"No recursive training rows after min_history_days={min_history_days}.")

    model = LGBMRegressor(**get_lgbm_params(args))
    model.fit(X_train, y_train)

    return model, X_train.columns.tolist(), X_train, y_train


def train_anchor_residual_lgbm(
    train_df: pd.DataFrame,
    target: str,
    base_date: pd.Timestamp,
    args: argparse.Namespace,
    min_history_days: int = 730,
):
    LGBMRegressor = import_lightgbm()

    train_df = train_df.sort_values("Date").reset_index(drop=True).copy()
    X = make_anchor_features(
        dates=train_df["Date"],
        history_df=train_df,
        target=target,
        base_date=base_date,
    )

    y = train_df[target].astype(float).reset_index(drop=True)
    first_train_date = train_df["Date"].min() + pd.Timedelta(days=min_history_days)

    mask = (
        (train_df["Date"] >= first_train_date).reset_index(drop=True)
        & X["anchor_revenue"].notna()
        & (X["anchor_revenue"] > 0)
    )

    y_residual = np.log1p(y.loc[mask]) - np.log1p(X.loc[mask, "anchor_revenue"])

    X_train = X.loc[mask].copy()
    y_train = y_residual.copy()

    if len(X_train) == 0:
        raise ValueError(f"No anchor training rows after min_history_days={min_history_days}.")

    model = LGBMRegressor(**get_lgbm_params(args))
    model.fit(X_train, y_train)

    return model, X_train.columns.tolist(), X_train, y_train


def clip_prediction(pred: float, history_values: pd.Series) -> float:
    history_values = history_values.astype(float)
    lower = 0.0
    upper = history_values.quantile(0.995) * 1.15

    if pd.isna(upper) or upper <= 0:
        upper = history_values.max() * 1.20

    return float(np.clip(pred, lower, upper))


def recursive_forecast_lgbm(
    model,
    history_df: pd.DataFrame,
    future_dates: Iterable[pd.Timestamp],
    target: str,
    feature_cols: List[str],
    base_date: pd.Timestamp,
    args: argparse.Namespace,
) -> np.ndarray:
    hist = history_df[["Date", target]].copy()
    hist = hist.sort_values("Date").set_index("Date")[target].astype(float)
    original_history_values = hist.copy()
    preds = []

    for d in pd.to_datetime(future_dates):
        d = pd.Timestamp(d)
        X_one = make_one_step_features(
            history_series=hist,
            date=d,
            target=target,
            feature_cols=feature_cols,
            base_date=base_date,
        )

        model_output = model.predict(X_one)[0]

        if args.target_mode == "yoy_ratio":
            lag_col = f"{target}_lag_365"
            lag_365 = X_one[lag_col].iloc[0]
            if pd.isna(lag_365) or lag_365 <= 0:
                pred_log = model_output
            else:
                pred_log = model_output + np.log1p(lag_365)
        else:
            pred_log = model_output

        pred = float(np.expm1(pred_log))
        pred = clip_prediction(pred, original_history_values)
        pred = max(pred, 0.0)

        preds.append(pred)
        hist.loc[d] = pred

    return np.asarray(preds, dtype=float)


def predict_anchor_residual_lgbm(
    model,
    history_df: pd.DataFrame,
    future_dates: Iterable[pd.Timestamp],
    target: str,
    feature_cols: List[str],
    base_date: pd.Timestamp,
) -> np.ndarray:
    X_future = make_anchor_features(
        dates=future_dates,
        history_df=history_df,
        target=target,
        base_date=base_date,
    ).reindex(columns=feature_cols)

    pred_residual = model.predict(X_future)
    pred_log = pred_residual + np.log1p(X_future["anchor_revenue"])
    pred = np.expm1(pred_log)

    upper = history_df[target].astype(float).quantile(0.995) * 1.15
    if pd.isna(upper) or upper <= 0:
        upper = history_df[target].astype(float).max() * 1.20

    pred = np.clip(pred, 0, upper)
    return np.asarray(pred, dtype=float)


def ensemble_predictions(
    pred_recursive: np.ndarray,
    pred_anchor: np.ndarray,
    pred_baseline: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    wr, wa, wb = normalize_weights(args.recursive_weight, args.anchor_weight, args.baseline_weight)
    pred = wr * pred_recursive + wa * pred_anchor + wb * pred_baseline
    return np.maximum(pred, 0.0)


def run_cv(data: LoadedData, args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    print("Running rolling time-series validation...")

    horizon = len(data.sample)
    fold_train_ends = ["2019-06-30", "2020-06-30", "2021-06-30"]
    rows = []
    sales_indexed = data.sales.set_index("Date")

    wr, wa, wb = normalize_weights(args.recursive_weight, args.anchor_weight, args.baseline_weight)

    for train_end_str in fold_train_ends:
        train_end = pd.Timestamp(train_end_str)
        valid_dates = pd.date_range(train_end + pd.Timedelta(days=1), periods=horizon, freq="D")

        if valid_dates.max() > data.sales["Date"].max():
            continue

        print(f"  Fold target=Revenue, train_end={train_end_str}")
        train_part = data.sales[data.sales["Date"] <= train_end].copy()
        y_true = sales_indexed.loc[valid_dates, TARGET].values

        pred_base = seasonal_naive_recursive(train_part, valid_dates, TARGET, use_trend=True)
        rows.append({
            "target": TARGET,
            "fold_train_end": train_end_str,
            "model": "SeasonalNaive365Trend",
            **regression_metrics(y_true, pred_base),
        })

        recursive_model, recursive_cols, _, _ = train_recursive_lgbm(
            train_df=train_part,
            target=TARGET,
            base_date=data.base_date,
            args=args,
        )
        pred_recursive = recursive_forecast_lgbm(
            model=recursive_model,
            history_df=train_part,
            future_dates=valid_dates,
            target=TARGET,
            feature_cols=recursive_cols,
            base_date=data.base_date,
            args=args,
        )
        rows.append({
            "target": TARGET,
            "fold_train_end": train_end_str,
            "model": f"RecursiveLightGBM_{args.target_mode}",
            **regression_metrics(y_true, pred_recursive),
        })

        anchor_model, anchor_cols, _, _ = train_anchor_residual_lgbm(
            train_df=train_part,
            target=TARGET,
            base_date=data.base_date,
            args=args,
        )
        pred_anchor = predict_anchor_residual_lgbm(
            model=anchor_model,
            history_df=train_part,
            future_dates=valid_dates,
            target=TARGET,
            feature_cols=anchor_cols,
            base_date=data.base_date,
        )
        rows.append({
            "target": TARGET,
            "fold_train_end": train_end_str,
            "model": "AnchorResidualLightGBM",
            **regression_metrics(y_true, pred_anchor),
        })

        pred_ens = ensemble_predictions(pred_recursive, pred_anchor, pred_base, args)
        rows.append({
            "target": TARGET,
            "fold_train_end": train_end_str,
            "model": f"Ensemble_R{wr:.2f}_A{wa:.2f}_B{wb:.2f}",
            **regression_metrics(y_true, pred_ens),
        })

    cv_results = pd.DataFrame(rows)
    cv_path = output_dir / "cv_results.csv"
    cv_results.to_csv(cv_path, index=False)

    cv_summary = cv_results.groupby(["target", "model"])[["MAE", "RMSE", "R2"]].mean().reset_index()
    cv_summary_path = output_dir / "cv_summary.csv"
    cv_summary.to_csv(cv_summary_path, index=False)

    print("Saved CV results:")
    print(f"  {cv_path}")
    print(f"  {cv_summary_path}")
    print(cv_summary)

    return cv_results


def save_feature_importance(model, feature_cols: List[str], model_name: str, output_dir: Path) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)

    safe_name = model_name.lower().replace(" ", "_")
    csv_path = output_dir / f"feature_importance_{safe_name}.csv"
    imp.to_csv(csv_path, index=False)

    top = imp.head(25).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.title(f"Top 25 Feature Importances - {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    png_path = output_dir / f"feature_importance_{safe_name}.png"
    plt.savefig(png_path, dpi=160)
    plt.close()

    print(f"Saved feature importance for {model_name}: {csv_path}, {png_path}")
    print(imp.head(10).to_string(index=False))
    return imp


def save_forecast_plots(sales: pd.DataFrame, submission: pd.DataFrame, output_dir: Path) -> None:
    sub_plot = submission.sort_values("Date")

    for target in ["Revenue", "COGS"]:
        plt.figure(figsize=(14, 5))
        plt.plot(sales["Date"], sales[target], label=f"Train {target}")
        plt.plot(sub_plot["Date"], sub_plot[target], label=f"Forecast {target}")
        plt.title(f"{target}: Train vs Forecast")
        plt.xlabel("Date")
        plt.ylabel(target)
        plt.legend()
        plt.tight_layout()

        out_path = output_dir / f"forecast_{target.lower()}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"Saved plot: {out_path}")


def maybe_save_shap(model, X_train: pd.DataFrame, output_dir: Path) -> None:
    try:
        import shap
    except ImportError:
        print("SHAP is not installed. To use it, run: pip install shap")
        return

    X_sample = X_train.sample(min(1000, len(X_train)), random_state=SEED)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    out_path = output_dir / "shap_summary_anchor_revenue.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP plot: {out_path}")


def make_submission(
    data: LoadedData,
    revenue_preds: np.ndarray,
    cogs_preds: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    pred_df = pd.DataFrame({
        "Date": data.future_dates_chrono,
        "Revenue": revenue_preds,
        "COGS": cogs_preds,
    })
    pred_map = pred_df.set_index("Date")

    submission = data.sample.copy()
    submission["Revenue"] = submission["Date"].map(pred_map["Revenue"])
    submission["COGS"] = submission["Date"].map(pred_map["COGS"])
    submission["Revenue"] = np.maximum(submission["Revenue"], 0).round(2)
    submission["COGS"] = np.maximum(submission["COGS"], 0).round(2)

    validate_submission(submission, data.sample, data.sample_original_order)
    submission.to_csv(output_path, index=False)
    print(f"Saved submission: {output_path}")

    return submission


def train_final_and_submit(
    data: LoadedData,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[pd.DataFrame, object, List[str], pd.DataFrame]:
    print("Training final recursive LightGBM model for Revenue...")
    recursive_model, recursive_cols, recursive_X_train, _ = train_recursive_lgbm(
        train_df=data.sales,
        target=TARGET,
        base_date=data.base_date,
        args=args,
    )
    pred_recursive = recursive_forecast_lgbm(
        model=recursive_model,
        history_df=data.sales,
        future_dates=data.future_dates_chrono,
        target=TARGET,
        feature_cols=recursive_cols,
        base_date=data.base_date,
        args=args,
    )

    print("Training final direct seasonal-anchor LightGBM model for Revenue...")
    anchor_model, anchor_cols, anchor_X_train, _ = train_anchor_residual_lgbm(
        train_df=data.sales,
        target=TARGET,
        base_date=data.base_date,
        args=args,
    )
    pred_anchor = predict_anchor_residual_lgbm(
        model=anchor_model,
        history_df=data.sales,
        future_dates=data.future_dates_chrono,
        target=TARGET,
        feature_cols=anchor_cols,
        base_date=data.base_date,
    )

    pred_baseline = seasonal_naive_recursive(
        history_df=data.sales,
        future_dates=data.future_dates_chrono,
        target=TARGET,
        use_trend=True,
    )
    pred_cogs = seasonal_naive_recursive(
        history_df=data.sales,
        future_dates=data.future_dates_chrono,
        target="COGS",
        use_trend=True,
    )
    pred_ensemble = ensemble_predictions(pred_recursive, pred_anchor, pred_baseline, args)

    components = pd.DataFrame({
        "Date": data.future_dates_chrono,
        "Revenue_recursive": pred_recursive,
        "Revenue_anchor": pred_anchor,
        "Revenue_baseline": pred_baseline,
        "Revenue_ensemble": pred_ensemble,
        "COGS": pred_cogs,
    })
    comp_path = output_dir / "prediction_components.csv"
    components.to_csv(comp_path, index=False)
    print(f"Saved prediction components: {comp_path}")

    make_submission(data, pred_baseline, pred_cogs, output_dir / "submission_baseline.csv")
    make_submission(data, pred_recursive, pred_cogs, output_dir / "submission_recursive_yoy.csv")
    make_submission(data, pred_ensemble, pred_cogs, output_dir / "submission_ensemble.csv")

    # Main Kaggle submission file: direct seasonal-anchor LightGBM.
    # This replaces the old submission_anchor_residual.csv name.
    final_submission = make_submission(data, pred_anchor, pred_cogs, output_dir / "submission.csv")

    print(f"Saved main recommended submission: {output_dir / 'submission.csv'}")

    return final_submission, anchor_model, anchor_cols, anchor_X_train


def main() -> int:
    args = parse_args()
    wr, wa, wb = normalize_weights(args.recursive_weight, args.anchor_weight, args.baseline_weight)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Datathon 2026 forecasting pipeline")
    print(f"Data directory: {data_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Recursive target mode: {args.target_mode}")
    print(f"Ensemble weights: recursive={wr:.2f}, anchor={wa:.2f}, baseline={wb:.2f}")

    data = load_and_validate(data_dir)

    if args.run_cv:
        run_cv(data, args, output_dir)
    else:
        print("Skipping CV. Add --run-cv if you need validation metrics for the report.")

    submission, anchor_model, anchor_cols, anchor_X_train = train_final_and_submit(
        data=data,
        args=args,
        output_dir=output_dir,
    )

    save_feature_importance(
        model=anchor_model,
        feature_cols=anchor_cols,
        model_name="anchor_revenue",
        output_dir=output_dir,
    )
    save_forecast_plots(sales=data.sales, submission=submission, output_dir=output_dir)

    if args.run_shap:
        maybe_save_shap(model=anchor_model, X_train=anchor_X_train, output_dir=output_dir)

    print("Done.")
    print("Main file to submit on Kaggle:")
    print(f"  {output_dir / 'submission.csv'}")
    print("\nOther submission files generated for reference:")
    print(f"  {output_dir / 'submission_ensemble.csv'}")
    print(f"  {output_dir / 'submission_recursive_yoy.csv'}")
    print(f"  {output_dir / 'submission_baseline.csv'}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nERROR:", exc, file=sys.stderr)
        raise
