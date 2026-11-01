
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "submissions"

HORIZONS = [36, 40, 44, 60, 64, 68]
QUANTILES = [0.025, 0.25, 0.50, 0.75, 0.975]
QUANTILE_COLUMNS = [
    "q0.025", "q0.25", "q0.5", "q0.75", "q0.975"
]


def load_weather(filename, value_column):
    path = RAW / filename

    df = pd.read_csv(
        path,
        sep=";",
        skipinitialspace=True,
        na_values=["-999", "-999.0"],
    )
    df.columns = df.columns.str.strip()

    df["timestamp"] = pd.to_datetime(
        df["MESS_DATUM"].astype(str).str.strip().str.zfill(10),
        format="%Y%m%d%H",
        utc=True,
    )

    # DWD timestamps are UTC; forecast timestamps are German local time.
    df["timestamp"] = (
    df["timestamp"]
    .dt.tz_convert("Europe/Berlin")
    .dt.tz_localize(None)
)

    df[value_column] = pd.to_numeric(
        df[value_column], errors="coerce"
    )

    df = df[["timestamp", value_column]].dropna()
    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp", keep="last")

    return df


def make_features(timestamps, reference_time):
    """Calendar features for the simple linear regression baseline."""
    t = pd.DatetimeIndex(timestamps)

    hour = t.hour + t.minute / 60
    weekday = t.dayofweek
    day_of_year = t.dayofyear

    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
        "year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        "trend_days": (
            (t - pd.Timestamp(reference_time))
            / pd.Timedelta(days=1)
        ).to_numpy(),
    })


def forecast_weather(df, value_column, target_times, is_wind=False):
    """Fit LR, estimate residual quantiles on a chronological holdout."""
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            f"Not enough observations for {value_column}: {len(df)}"
        )

    X = make_features(df["timestamp"], df["timestamp"].iloc[0])
    y = df[value_column].to_numpy()

    # Keep the most recent 20% for estimating forecast errors.
    split = int(len(df) * 0.8)
    if split < 50 or len(df) - split < 20:
        raise ValueError("Not enough data for chronological validation.")

    validation_model = LinearRegression()
    validation_model.fit(X.iloc[:split], y[:split])

    validation_predictions = validation_model.predict(X.iloc[split:])
    residuals = y[split:] - validation_predictions

    error_quantiles = np.quantile(residuals, QUANTILES)

    # Refit using all observations for the final forecast.
    model = LinearRegression()
    model.fit(X, y)

    target_X = make_features(target_times, df["timestamp"].iloc[0])
    point_forecasts = model.predict(target_X)

    forecasts = []
    for point in point_forecasts:
        values = point + error_quantiles

        if is_wind:
            values = np.maximum(values, 0)

        # Ensure the quantiles are nondecreasing.
        values = np.maximum.accumulate(values)
        forecasts.append(values)

    return np.asarray(forecasts)


def load_energy():
    path = RAW / "energy_data_2020_2026.csv"
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")

    for col in ["load", "residual_load"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["date", "hour", "load", "residual_load"]
    ).copy()

    # In this dataset, hour=1 represents 00:00, hour=2 represents
    # 01:00, etc. Verify this convention against the seminar materials.
    df["timestamp"] = (
        df["date"] + pd.to_timedelta(df["hour"] - 1, unit="h")
    )

    df = df.sort_values("timestamp")
    return df


def forecast_energy(df, target_times, value_column):
    forecasts = []

    for target in target_times:
        # Match the target's weekday and clock hour, using only
        # observations earlier than the target timestamp.
        history = df[
            (df["timestamp"] < target)
            & (df["timestamp"].dt.dayofweek == target.dayofweek)
            & (df["timestamp"].dt.hour == target.hour)
        ].sort_values("timestamp").tail(20)

        if len(history) < 5:
            raise ValueError(
                f"Only {len(history)} matching historical observations "
                f"for {value_column} at {target}"
            )

        values = np.quantile(history[value_column], QUANTILES)
        values = np.maximum.accumulate(values)
        forecasts.append(values)

    return np.asarray(forecasts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forecast-date",
        default="2026-10-28",
        help="Wednesday forecast date, YYYY-MM-DD",
    )
    parser.add_argument(
        "--identifier",
        default="sample",
        help="Submission identifier",
    )
    args = parser.parse_args()

    forecast_date = pd.Timestamp(args.forecast_date)
    if forecast_date.dayofweek != 2:
        raise ValueError("The forecast date should be a Wednesday.")

    # Horizons are measured from the start of Thursday, following
    # the Wednesday submission deadline.
    origin = forecast_date + pd.Timedelta(days=1)
    target_times = [origin + pd.Timedelta(hours=h) for h in HORIZONS]

    print("Target timestamps:")
    for horizon, timestamp in zip(HORIZONS, target_times):
        print(f"  {horizon} hour -> {timestamp}")

    temperature = load_weather(
        "produkt_tu_stunde_20250418_20261019_01420.txt",
        "TT_TU",
    )
    wind = load_weather(
        "produkt_ff_stunde_20250418_20261019_01420.txt",
        "F",
    )

    temperature_forecasts = forecast_weather(
        temperature, "TT_TU", target_times
    )
    wind_forecasts = forecast_weather(
        wind, "F", target_times, is_wind=True
    )

    energy = load_energy()
    demand_forecasts = forecast_energy(
        energy, target_times, "load"
    )
    residual_forecasts = forecast_energy(
        energy, target_times, "residual_load"
    )

    all_forecasts = {
        "demand": demand_forecasts,
        "res_demand": residual_forecasts,
        "temperature": temperature_forecasts,
        "wind": wind_forecasts,
    }

    rows = []
    for target_name, forecasts in all_forecasts.items():
        for horizon, values in zip(HORIZONS, forecasts):
            row = {
                "forecast_date": forecast_date.strftime("%Y-%m-%d"),
                "target": target_name,
                "horizon": f"{horizon} hour",
            }
            row.update(dict(zip(QUANTILE_COLUMNS, values)))
            rows.append(row)

    columns = [
        "forecast_date",
        "target",
        "horizon",
        *QUANTILE_COLUMNS,
    ]
    result = pd.DataFrame(rows, columns=columns)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / (
        f"{forecast_date:%Y%m%d}_{args.identifier}.csv"
    )
    result.to_csv(output_path, index=False, float_format="%.5f")

    print(f"\nSaved {len(result)} forecasts to:")
    print(output_path)
    print("\nPreview:")
    print(result.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
