import pandas as pd
import re
import requests
import json
import time

def clean_price(value):
    """Convert a price string like '72,800' or 'Price not found' to int."""
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return 0
    value = value.strip()
    if "not found" in value.lower() or value == "0" or value == "":
        return 0
    cleaned = re.sub(r"[^0-9]", "", value)
    return int(cleaned) if cleaned else 0


def parse_name_from_title(title: str):
    """
    Attempt to extract first and last name from vehicle title.
    If not possible, uses default values.
    """
    # Try to extract name from title (e.g., "John's 2023 Dodge" -> "John")
    # This is a simple heuristic - adjust based on your actual data format
    words = title.split()
    if len(words) > 0:
        # Check if first word looks like a name (starts with capital, not a number)
        first_word = words[0].strip("'s").strip("'")
        if first_word and first_word[0].isupper() and not first_word[0].isdigit():
            # Try to split into first/last if there's an apostrophe or comma
            if "'" in first_word:
                name_part = first_word.split("'")[0]
                return name_part, "Owner"
            return first_word, "Owner"
    
    # Default values if name cannot be extracted
    return "Vehicle", "Owner"


def clean_mileage(mileage):
    """Convert mileage to integer, handling string formats."""
    if isinstance(mileage, (int, float)):
        return str(int(mileage))
    if not isinstance(mileage, str):
        return "0"
    # Remove commas and non-numeric characters except digits
    cleaned = re.sub(r"[^0-9]", "", str(mileage))
    return cleaned if cleaned else "0"


def send_lead_to_salesforce(vin: str, mileage: str, vehicle_name: str):
    """
    Send lead data to Salesforce API endpoint.
    
    Args:
        vin: Vehicle VIN number
        mileage: Vehicle mileage
        vehicle_name: Vehicle name/title
        
    Returns:
        bool: True if successful, False otherwise
    """
    url = "https://iframe-backend.vercel.app/api/submit-chatbot-form"
    
    # Parse name from vehicle title
    first_name, last_name = parse_name_from_title(vehicle_name)
    
    # Prepare JSON payload in required format
    payload = {
        "contact": {
            "first_name": first_name,
            "last_name": last_name,
            "email": "",  # Not available in current data
            "phone": ""   # Not available in current data
        },
        "vehicle": {
            "vin": vin,
            "mileage": mileage
        }
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"  ✓ Successfully sent to Salesforce")
            return True
        else:
            print(f"  ✗ Failed: HTTP {response.status_code} - {response.text[:100]}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error sending to Salesforce: {str(e)}")
        return False


def compare_prices(input_file: str, output_file: str = None):
    """
    Compare original vs Adjusted MMR Price and send leads to Salesforce where price < MMR.
    
    Args:
        input_file: Path to input CSV file with MMR data
        output_file: Optional path for CSV backup (deprecated, kept for compatibility)
    """
    df = pd.read_csv(input_file)

    required_cols = ["Vehicle Name", "VIN", "Mileage", "Original Price", "Adjusted MMR Price"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    leads_sent = 0
    leads_failed = 0

    for _, row in df.iterrows():
        title = str(row["Vehicle Name"]).strip()
        vin = str(row["VIN"]).strip()
        mileage = clean_mileage(row["Mileage"])
        original_price = clean_price(row["Original Price"])
        adjusted_price = clean_price(row["Adjusted MMR Price"])

        # Skip invalid or missing values
        if adjusted_price == 0 or original_price == 0:
            print(f"[SKIP] Missing or invalid price for {title}")
            continue

        if original_price < adjusted_price:
            print(f"[LEAD FOUND] {title}")
            print(f"  Original: ${original_price:,} | MMR: ${adjusted_price:,}")
            print(f"  VIN: {vin} | Mileage: {mileage}")
            
            # Send to Salesforce instead of saving to CSV
            if send_lead_to_salesforce(vin, mileage, title):
                leads_sent += 1
            else:
                leads_failed += 1
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
        else:
            print(f"[LEAD LOST] {title} — Original: ${original_price:,} | MMR: ${adjusted_price:,}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Leads sent to Salesforce: {leads_sent}")
    if leads_failed > 0:
        print(f"Leads failed: {leads_failed}")
    if leads_sent == 0 and leads_failed == 0:
        print("No leads found — all vehicles overpriced.")
    print(f"{'='*60}")


if __name__ == "__main__":
    input_csv = "mmr_prices.csv"  # Input file with MMR data
    # Note: output_csv parameter is kept for compatibility but no longer used
    # Data is now sent directly to Salesforce API
    compare_prices(input_csv)
