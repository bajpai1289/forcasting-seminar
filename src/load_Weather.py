
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

temp_path = DATA_DIR / (
    ""
    "produkt_tu_stunde_20250418_20261019_01420.txt"
)

wind_path = DATA_DIR / (
    ""
    "produkt_ff_stunde_20250418_20261019_01420.txt"
)


def inspect_dwd_file(path):
    # DWD hourly observation files are typically semicolon-separated.
    df = pd.read_csv(path, sep=";", na_values=["-999"])

    print(f"\nFile: {path}")
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print(df.head())
    print(df.isna().sum())

    return df


if __name__ == "__main__":
    temp = inspect_dwd_file(temp_path)
    wind = inspect_dwd_file(wind_path)
