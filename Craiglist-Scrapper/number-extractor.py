import csv
import re
from typing import List, Dict

def extract_phone_numbers(text: str) -> List[str]:
    """
    Extract phone numbers from text in various formats with validation.
    Supports formats like:
    - (555) 555-5555
    - 555-555-5555
    - 555.555.5555
    - 5555555555
    - 555 555 5555
    
    Args:
        text: Text to extract phone numbers from
        
    Returns:
        List of validated US phone numbers
    """
    if not text:
        return []
    
    # Pattern to match US phone numbers with word boundaries to avoid false positives
    # Matches formats: (XXX) XXX-XXXX, XXX-XXX-XXXX, XXX.XXX.XXXX, XXX XXX XXXX, or 10 digits
    patterns = [
        r'\b\(?([2-9]\d{2})\)?[-.\s]?([2-9]\d{2})[-.\s]?(\d{4})\b',  # Standard formats with validation
        r'\b([2-9]\d{2})[-.\s]?([2-9]\d{2})[-.\s]?(\d{4})\b',        # Without parentheses
    ]
    
    phone_numbers = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            # Reconstruct phone number in a consistent format
            area_code = match.group(1)
            exchange = match.group(2)
            number = match.group(3)
            formatted_phone = f"{area_code}-{exchange}-{number}"
            phone_numbers.append(formatted_phone)
    
    # Also check for standalone 10-digit numbers (with word boundaries)
    # This is more restrictive and validates format
    standalone_pattern = r'\b([2-9]\d{2})([2-9]\d{2})(\d{4})\b'
    standalone_matches = re.finditer(standalone_pattern, text)
    for match in standalone_matches:
        area_code = match.group(1)
        exchange = match.group(2)
        number = match.group(3)
        formatted_phone = f"{area_code}-{exchange}-{number}"
        phone_numbers.append(formatted_phone)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_phones = []
    for phone in phone_numbers:
        # Clean the phone number for comparison (remove formatting)
        cleaned = re.sub(r'[^\d]', '', phone)
        if cleaned not in seen and len(cleaned) == 10:
            seen.add(cleaned)
            unique_phones.append(phone)
    
    return unique_phones

def process_csv(input_file: str, output_file: str = None) -> List[Dict]:
    """
    Read CSV file and extract phone numbers from 'Vehicle Details' column.
    
    Args:
        input_file: Path to input CSV file
        output_file: Optional path to output CSV file with extracted numbers
    
    Returns:
        List of dictionaries containing original data and extracted phone numbers
    """
    results = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Find the vehicle details column (case-insensitive search)
            vehicle_details_column = None
            for col in reader.fieldnames:
                if col.lower() == 'vehicle details':
                    vehicle_details_column = col
                    break
            
            if not vehicle_details_column:
                print(f"Warning: 'Vehicle Details' column not found.")
                print(f"Available columns: {reader.fieldnames}")
                return results
            
            for row in reader:
                vehicle_details = row.get(vehicle_details_column, '')
                phone_numbers = extract_phone_numbers(vehicle_details)
                
                result = {
                    'original_data': vehicle_details,
                    'extracted_phones': phone_numbers,
                    'phone_count': len(phone_numbers)
                }
                
                # Add all other columns from the original CSV
                result.update({k: v for k, v in row.items() if k != vehicle_details_column})
                
                results.append(result)
                
                # Print extracted phone numbers
                if phone_numbers:
                    print(f"Found {len(phone_numbers)} phone number(s): {', '.join(phone_numbers)}")
                else:
                    print("No phone numbers found in this row")
        
        # Optionally write results to a new CSV
        if output_file:
            write_results_to_csv(results, output_file)
            print(f"\nResults written to: {output_file}")
        
        return results
    
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return []
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return []

def write_results_to_csv(results: List[Dict], output_file: str):
    """Write extraction results to a new CSV file with only phone numbers column."""
    if not results:
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Extracted Phone Numbers'])
        
        # Write phone numbers (one per row)
        for result in results:
            phone_numbers = result['extracted_phones']
            if phone_numbers:
                # Join multiple phone numbers with comma if multiple found
                writer.writerow([', '.join(phone_numbers)])
            else:
                # Write empty cell if no phone number found
                writer.writerow([''])

# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    input_csv = "bakersfield_craigslist_detailed_20251105_012203.csv"
    output_csv = "extracted_phones.csv"
    
    print("Starting phone number extraction...\n")
    results = process_csv(input_csv, output_csv)
    
    print(f"\n{'='*50}")
    print(f"Total rows processed: {len(results)}")
    print(f"Rows with phone numbers: {sum(1 for r in results if r['phone_count'] > 0)}")
    print(f"Total phone numbers found: {sum(r['phone_count'] for r in results)}")