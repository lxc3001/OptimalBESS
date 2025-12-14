import h5py
import numpy as np
import pandas as pd
import argparse
from pandas.tseries.frequencies import to_offset


def load_forecast_time_and_mean(path: str) -> pd.DataFrame:
    """Load only forecast target timestamps and a single forecast value (mean across members)."""
    with h5py.File(path, "r") as f:
        forecast_time_raw = f["forecast_time"][:]
        forecast_time = pd.to_datetime([s.decode("utf-8") for s in forecast_time_raw])

        forecasts = np.asarray(f["forecasts"][:])
        forecast_value_mean = forecasts.mean(axis=1)

    return pd.DataFrame({"forecast_time": forecast_time, "forecast_value": forecast_value_mean})


def load_actual_time_and_value(path: str) -> pd.DataFrame:
    """Load only actual timestamps and actual values."""
    with h5py.File(path, "r") as f:
        time_index_raw = f["time_index"][:]
        time_index = pd.to_datetime([
            (s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)) for s in time_index_raw
        ])
        actual_value = np.asarray(f["actuals"][:])

    return pd.DataFrame({"time": time_index, "actual_value": actual_value})


def merge_forecast_and_actuals_csv(
    forecast_csv_path: str,
    actuals_csv_path: str,
    freq: str = "1H",
    actuals_agg: str = "mean",
    fill_missing: bool = False,
    fill_actual_method: str = "time",
) -> pd.DataFrame:
    """Merge forecast (hourly) and actuals (5-min) by time, dropping rows with any zeros.

    - Forecast CSV columns expected: forecast_time, forecast_value
    - Actuals CSV columns expected: time, actual_value
    - Forecast is upsampled to `freq` using time-based linear interpolation.
    - Actuals are resampled to `freq` using `actuals_agg`.
    """
    forecast_df = pd.read_csv(forecast_csv_path)
    actuals_df = pd.read_csv(actuals_csv_path)

    forecast_df["forecast_time"] = pd.to_datetime(forecast_df["forecast_time"])
    actuals_df["time"] = pd.to_datetime(actuals_df["time"])

    # Forecast: hourly -> freq (e.g., 15min) via time-based linear interpolation.
    # Treat zeros as missing before interpolation.
    forecast_series = (
        forecast_df.set_index("forecast_time")["forecast_value"]
        .sort_index()
        .astype(float)
    )
    forecast_series = forecast_series.replace(0, np.nan)
    offset = to_offset(freq)
    # Extend the end by (1H - offset) so we cover 23:15/23:30/23:45 when freq=15min.
    end = forecast_series.index.max() + pd.Timedelta(hours=1) - offset
    full_index = pd.date_range(
        start=forecast_series.index.min(),
        end=end,
        freq=freq,
        tz=forecast_series.index.tz,
    )
    forecast_resampled = forecast_series.reindex(full_index)
    forecast_resampled = forecast_resampled.interpolate(method="time").ffill().bfill()

    # Actuals: 5-min -> freq (e.g., 15min) using aggregation.
    # Treat zeros as missing before resampling so they don't bias mean.
    actuals_series = (
        actuals_df.set_index("time")["actual_value"].sort_index().astype(float).replace(0, np.nan)
    )
    resampler = actuals_series.resample(freq)
    if actuals_agg == "mean":
        actuals_resampled = resampler.mean()
    elif actuals_agg == "first":
        actuals_resampled = resampler.first()
    elif actuals_agg == "last":
        actuals_resampled = resampler.last()
    else:
        raise ValueError("actuals_agg must be one of: mean, first, last")
    actuals_resampled = actuals_resampled.rename("actual_value")

    if fill_missing:
        if fill_actual_method not in {"time", "ffill"}:
            raise ValueError("fill_actual_method must be one of: time, ffill")
        actuals_resampled = actuals_resampled.reindex(full_index)
        if fill_actual_method == "time":
            actuals_resampled = actuals_resampled.interpolate(method="time")
        actuals_resampled = actuals_resampled.ffill().bfill()

        merged = pd.DataFrame(
            {
                "forecast_time": full_index,
                "forecast_value": forecast_resampled.values,
                "actual_value": actuals_resampled.values,
            }
        )
    else:
        forecast_df_resampled = (
            forecast_resampled.rename("forecast_value").to_frame().reset_index(names="time")
        )
        actuals_df_resampled = actuals_resampled.reset_index().rename(columns={"time": "time"})
        merged = forecast_df_resampled.merge(actuals_df_resampled, on="time", how="inner")
        merged = merged.rename(columns={"time": "forecast_time"})

    merged = merged.dropna(subset=["forecast_value", "actual_value", "forecast_time"])
    merged = merged[(merged["forecast_value"] != 0) & (merged["actual_value"] != 0)]

    merged = merged.sort_values("forecast_time").reset_index(drop=True)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Export wind data to CSV. "
            "Forecast: forecast_time + mean forecast_value. "
            "Actuals: time + actual_value."
        )
    )
    parser.add_argument(
        "--kind",
        choices=["forecast", "actuals", "merged"],
        default="forecast",
        help="Which dataset to export",
    )
    parser.add_argument(
        "--input",
        default="Site_amazon_wind_farm_texas_wind_day-ahead_fcst_2018.h5",
        help="Path to the forecast .h5 file",
    )
    parser.add_argument(
        "--output",
        default="wind_day_ahead_forecast_mean_2018.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--forecast_csv",
        default="wind_day_ahead_forecast_mean_2018.csv",
        help="Forecast CSV path (only used when --kind merged)",
    )
    parser.add_argument(
        "--actuals_csv",
        default="wind_actuals_2018.csv",
        help="Actuals CSV path (only used when --kind merged)",
    )
    parser.add_argument(
        "--actuals_agg",
        choices=["mean", "first", "last"],
        default="mean",
        help=(
            "How to aggregate actuals into the target frequency when merging. "
            "Use 'first' to match the original actual at each bucket start."
        ),
    )
    parser.add_argument(
        "--freq",
        default="1H",
        help="Target frequency when merging (e.g., 1H, 15min)",
    )
    parser.add_argument(
        "--fill_missing",
        action="store_true",
        help="If set, output a complete time grid at --freq and fill missing values.",
    )
    parser.add_argument(
        "--fill_actual_method",
        choices=["time", "ffill"],
        default="time",
        help="How to fill missing actuals after resampling (only with --fill_missing)",
    )
    args = parser.parse_args()

    if args.kind == "forecast":
        df = load_forecast_time_and_mean(args.input)
    elif args.kind == "actuals":
        df = load_actual_time_and_value(args.input)
    else:
        df = merge_forecast_and_actuals_csv(
            args.forecast_csv,
            args.actuals_csv,
            freq=args.freq,
            actuals_agg=args.actuals_agg,
            fill_missing=args.fill_missing,
            fill_actual_method=args.fill_actual_method,
        )
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")