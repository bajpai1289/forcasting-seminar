
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# The quantile calculation uses NumPy's method="linear"  because of the explicit quantile() function spicification.
QUANTILES = [0.025, 0.25, 0.50, 0.75, 0.975]

QUANTILE_COLUMNS = [
    "q0.025",
    "q0.25",
    "q0.5",
    "q0.75",
    "q0.975",
]

API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Approximate Frankfurt Airport coordinates.
LATITUDE = 50.05
LONGITUDE = 8.60

MODEL = "ecmwf_ifs025_ensemble"


def fetch_ensemble_forecast(target_times):
    """
    Download ECMWF IFS 0.25 ensemble forecasts
    and calculate quantiles across the 51 members.

    target_times must be UTC timestamps.
    """

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "models": MODEL,
        "hourly": "temperature_2m,wind_speed_10m",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "timezone": "GMT",
        "forecast_days": 7,
        "temporal_resolution": "hourly",
    }

    print("Requesting ECMWF ensemble forecasts...")

    response = requests.get(
        API_URL,
        params=params,
        timeout=90,
    )

    response.raise_for_status()
    data = response.json()

    if data.get("error"):
        raise RuntimeError(data.get("reason", "API error"))

    hourly = data["hourly"]

    forecast = pd.DataFrame(hourly)
    forecast["time"] = pd.to_datetime(
        forecast["time"], utc=True
    )

    print("API model:", MODEL)
    print("API timezone:", data.get("timezone"))
    print("API coordinates:", data.get("latitude"),
          data.get("longitude"))

    results = []

    variables = {
        "temperature": "temperature_2m",
        "wind": "wind_speed_10m",
    }

    for target_time in target_times:
        print("Here")
        target_time = pd.Timestamp(target_time)

        if target_time.tzinfo is None:
            target_time = target_time.tz_localize("UTC")
        else:
            target_time = target_time.tz_convert("UTC")

        matching = forecast[
            forecast["time"] == target_time
        ]

        if matching.empty:
            raise ValueError(
                f"Forecast timestamp unavailable: {target_time}"
            )

        row = {"timestamp": target_time}

        for target_name, variable in variables.items():

            # Identify all ensemble members dynamically.
            member_columns = [
                col for col in forecast.columns
                if col.startswith(variable + "_member")
            ]

            if len(member_columns) != 51:
                raise ValueError(
                    f"Expected 51 members for {variable}, "
                    f"found {len(member_columns)}.\n"
                    f"Available columns: {member_columns}"
                )

            values = matching[member_columns].iloc[0].to_numpy(
                dtype=float
            )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Missing ensemble values for "
                    f"{variable} at {target_time}"
                )

            # NumPy's linear quantile method corresponds
            # to R quantile()'s default type 7.
            quantiles = np.quantile(
                values,
                QUANTILES,
                method="linear",
            )

            for column, value in zip(
                QUANTILE_COLUMNS, quantiles
            ):
                row[f"{target_name}_{column}"] = value

        results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":

    # Example forecast date: Wednesday 4 November 2026.
    # Weather target timestamps are in UTC.

    target_times = [
        "2026-11-06 12:00",
        "2026-11-06 16:00",
        "2026-11-06 20:00",
        "2026-11-07 12:00",
        "2026-11-07 16:00",
        "2026-11-07 20:00",
    ]

    result = fetch_ensemble_forecast(target_times)

    print("\nWeather ensemble quantiles:")
    print(result.to_string(index=False))

    output_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "weather_ensemble_20261104.csv"
    )

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    result.to_csv(output_path, index=False)

    print("\nSaved to:", output_path)
