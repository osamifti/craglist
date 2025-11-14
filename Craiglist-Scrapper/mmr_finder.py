import os
import time
import csv
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === CONFIG ===
MANHEIM_ENDPOINT = os.getenv("MANHEIM_ENDPOINT")
MANHEIM_AUTH = os.getenv("MANHEIM_AUTH")
MANHEIM_API_BASE_URL = os.getenv("MANHEIM_API_BASE_URL")

_cached_token = None
_token_expiry = 0


def authenticate_manheim():
    """Authenticate with Manheim API using client credentials."""
    global _cached_token, _token_expiry

    if _cached_token and time.time() < _token_expiry:
        return _cached_token

    headers = {
        "Authorization": f"Basic {MANHEIM_AUTH}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}

    response = requests.post(MANHEIM_ENDPOINT, headers=headers, data=data)
    response.raise_for_status()

    token_data = response.json()
    _cached_token = token_data.get("access_token")
    expires_in = token_data.get("expires_in", 1800)
    _token_expiry = time.time() + expires_in - 60

    return _cached_token


def get_valuation_by_vin(vin: str, mileage: int):
    """Fetch valuation data for a vehicle by VIN and mileage."""
    try:
        token = authenticate_manheim()
        headers = {"Authorization": f"Bearer {token}"}
        url = f"{MANHEIM_API_BASE_URL}/valuations/vin/{vin}?grade=45&odometer={mileage}"

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        items = data.get("items", [])
        if not items:
            raise ValueError("No valuation data found")

        adjusted_price = items[0]["adjustedPricing"]["wholesale"]["average"]
        return adjusted_price

    except Exception as e:
        print(f"[ERROR] VIN {vin}: {e}")
        return "Price not found"


def process_vehicle_csv(input_file: str, output_file: str):
    """Reads VIN + mileage from CSV, fetches valuations, and writes output CSV."""
    df = pd.read_csv(input_file)

    # Prepare output rows
    results = []

    for _, row in df.iterrows():
        vin = str(row.get("VIN", "")).strip()
        mileage = str(row.get("Mileage", "")).replace(",", "").strip()
        title = str(row.get("Title", "Unknown Vehicle")).strip()
        original_price = str(row.get("Price", "")).replace("$", "").replace(",", "").strip()

        if not vin or vin.lower() == "not found":
            print(f"[SKIP] No VIN for '{title}'")
            continue

        try:
            mileage = int(mileage)
        except ValueError:
            print(f"[SKIP] Invalid mileage '{mileage}' for '{title}'")
            continue

        print(f"[INFO] Fetching price for {title} (VIN: {vin}) ...")
        adjusted_price = get_valuation_by_vin(vin, mileage)

        results.append({
            "Vehicle Name": title,
            "Mileage": mileage,
            "VIN": vin,
            "Adjusted MMR Price": adjusted_price,
            "Original Price": original_price
        })                                                                                                                                                                                                                                                                                                  

    # Write to new CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\nProcess complete! Results saved to '{output_file}'")
    print(f"Total processed: {len(results)} vehicles")


if __name__ == "__main__":
    input_csv = "bakersfield_craigslist_detailed_20251105_192012.csv"  # your input file
    output_csv = "mmr_prices.csv"  # your output file

    process_vehicle_csv(input_csv, output_csv)
