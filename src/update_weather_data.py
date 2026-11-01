from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import re

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

STATION = "01420"

DATASETS = {
    "temperature": {
        "product": "TU",
        "filename": "produkt_tu_stunde_20250418_20261019_01420.txt",
    },
    "wind": {
        "product": "FF",
        "filename": "produkt_ff_stunde_20250418_20261019_01420.txt",
    },
}

BASE_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/hourly"
)


def download_latest_dataset(product):
    """Download the latest available DWD station ZIP."""
    folder = (
        f"{BASE_URL}/"
        f"{'air_temperature' if product == 'TU' else 'wind'}/recent/"
    )

    response = requests.get(folder, timeout=30)
    response.raise_for_status()

    pattern = (
        rf'href=["\'](stundenwerte_{product}_{STATION}_akt\.zip)["\']'
    )
    matches = re.findall(pattern, response.text)

    if not matches:
        raise FileNotFoundError(
            f"Could not find the recent DWD ZIP for "
            f"{product}, station {STATION}"
        )

    zip_filename = matches[0]
    zip_url = folder + zip_filename

    print(f"Downloading {zip_filename}...")

    download = requests.get(zip_url, timeout=60)
    download.raise_for_status()

    return download.content


def read_product_from_zip(zip_bytes):
    """Read the product TXT file inside a DWD ZIP archive."""
    with ZipFile(BytesIO(zip_bytes)) as archive:
        txt_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".txt")
            and "produkt" in Path(name).name.lower()
        ]

        if not txt_files:
            raise ValueError("No product TXT file found in DWD ZIP.")

        with archive.open(txt_files[0]) as file:
            df = pd.read_csv(
                file,
                sep=";",
                skipinitialspace=True,
                na_values=["-999", "-999.0"],
                dtype={"MESS_DATUM": str},
            )

    df.columns = df.columns.str.strip()
    df["MESS_DATUM"] = (
        df["MESS_DATUM"].astype(str).str.strip().str.zfill(10)
    )

    return df


def update_dataset(name, config):
    product = config["product"]
    path = RAW / config["filename"]

    if not path.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {path}"
        )

    print(f"\nUpdating {name}...")

    # Load existing historical + previously downloaded observations.
    old = pd.read_csv(
        path,
        sep=";",
        skipinitialspace=True,
        na_values=["-999", "-999.0"],
        dtype={"MESS_DATUM": str},
    )
    old.columns = old.columns.str.strip()
    old["MESS_DATUM"] = (
        old["MESS_DATUM"].astype(str).str.strip().str.zfill(10)
    )

    old_latest = old["MESS_DATUM"].max()

    # Download the latest DWD observations.
    zip_bytes = download_latest_dataset(product)
    recent = read_product_from_zip(zip_bytes)

    recent_latest = recent["MESS_DATUM"].max()

    # Combine columns safely, even if DWD adds a new metadata column.
    combined = pd.concat(
        [old, recent],
        ignore_index=True,
        sort=False,
    )

    # Keep the latest version of any duplicate timestamp.
    combined = combined.drop_duplicates(
        subset=["MESS_DATUM"],
        keep="last",
    )

    combined = combined.sort_values("MESS_DATUM").reset_index(drop=True)

    if combined.empty:
        raise ValueError(f"{name} dataset is empty after merging.")

    # Preserve the existing input filename used by the forecast script.
    combined.to_csv(path, sep=";", index=False)

    print(f"Previous latest observation: {old_latest}")
    print(f"Latest downloaded observation: {recent_latest}")
    print(f"Updated latest observation: {combined['MESS_DATUM'].max()}")
    print(f"Total observations: {len(combined)}")
    print(f"Saved to: {path}")


def main():
    RAW.mkdir(parents=True, exist_ok=True)

    for name, config in DATASETS.items():
        update_dataset(name, config)

    print("\nDWD weather data update complete.")


if __name__ == "__main__":
    main()