# Complete Step-by-Step Analysis of `allrounder.py`

## Overview

`allrounder.py` is an **integrated end-to-end pipeline** that combines all functionalities into a single automated workflow:
1. Web Scraping → 2. Phone Extraction → 3. MMR Pricing → 4. Lead Qualification → 5. Salesforce Integration → 6. SMS Outreach

---

## System Architecture

### Configuration Class (`PipelineConfig`)
- Centralized configuration for all components
- Loads credentials from `.env` file
- Defines API endpoints, delays, and file paths

### Main Components:
1. **CraigslistScraper** - Web scraping
2. **PhoneExtractor** - Phone number extraction
3. **MMRPricingService** - Manheim API pricing
4. **LeadQualifier** - Lead qualification logic
5. **SalesforceIntegration** - Salesforce API submission
6. **SMSOutreach** - Twilio SMS sending
7. **DataExporter** - CSV export
8. **LeadGenerationPipeline** - Main orchestrator

---

## Complete Step-by-Step Workflow

### **PHASE 1: INITIALIZATION**

#### Step 1.1: Configuration Setup
```python
PipelineConfig class loads:
- Craigslist target URL
- Manheim API credentials (endpoint, auth, base URL)
- Salesforce API URL
- Twilio credentials (account SID, auth token, phone number)
- OpenAI API key
- Initial SMS message template
- Output file names (timestamped)
- Processing delays (API: 0.5s, scraping: 2s)
```

#### Step 1.2: Logging Setup
- Creates timestamped log file: `pipeline_YYYYMMDD_HHMMSS.log`
- Logs to both file and console
- Tracks all operations throughout the pipeline

#### Step 1.3: Validation
- Validates all required environment variables are present
- Raises error if any credentials are missing

---

### **PHASE 2: WEB SCRAPING** (`CraigslistScraper`)

#### Step 2.1: Browser Setup
- Initializes Chrome WebDriver with Selenium
- Configures browser options (headless mode optional)
- Sets up WebDriverWait for element waits
- Auto-downloads ChromeDriver using webdriver-manager

#### Step 2.2: Navigate to Craigslist
- Navigates to: `https://bakersfield.craigslist.org/search/cta?purveyor=owner`
- Waits **40 seconds** for page to fully load (dynamic content)

#### Step 2.3: Apply Filters
- **Owner Filter**: Clicks "owner" button to show only owner listings
- **Bundle Duplicates**: Clicks checkbox to bundle duplicate listings
- Waits **5 seconds** after filters are applied

#### Step 2.4: Discover Listings
- Finds all gallery cards using XPath: `//div[@class='gallery-card']`
- Counts total number of listings to process
- Logs total count

#### Step 2.5: Scrape Each Listing
For each gallery card:
1. **Re-fetch cards** (to avoid stale element references)
2. **Extract listing URL** before clicking
3. **Click gallery card** to open detail page
4. **Click "show-contact" button** (twice, 5 seconds apart) to reveal phone number
5. **Scrape vehicle details**:
   - Title (`//span[@id='titletextonly']`)
   - Price (`//span[@class='price']`)
   - Mileage (`//div[@class='attr auto_miles']/span[@class='valu']`)
   - VIN (`//div[@class='attr auto_vin']/span[@class='valu']`)
   - Vehicle Details (`//section[@id='postingbody']`)
6. **Add metadata**: listing URL, scraped timestamp
7. **Navigate back** to gallery (handles double-back if contact modal opened)
8. **Wait 2 seconds** before next listing

#### Step 2.6: Output
- Returns list of vehicle dictionaries
- Each vehicle contains: `title`, `price`, `mileage`, `vin`, `vehicle_details`, `listing_url`, `scraped_at`

---

### **PHASE 3: PHONE NUMBER EXTRACTION** (`PhoneExtractor`)

#### Step 3.1: Extract Phone Numbers
For each vehicle's `vehicle_details` text:
- Uses regex patterns to find US phone numbers
- Supports formats: `(555) 555-5555`, `555-555-5555`, `555.555.5555`, `555 555 5555`, `5555555555`
- Validates area code and exchange (must start with 2-9)
- Formats as: `XXX-XXX-XXXX`

#### Step 3.2: Add to Vehicle Data
- Adds `phone_numbers` list to each vehicle dictionary
- Adds `phone_count` (number of phones found)
- Logs extraction results

#### Step 3.3: Output
- Returns vehicles with phone number data added
- Tracks total vehicles with phone numbers found

---

### **PHASE 4: MMR PRICING** (`MMRPricingService`)

#### Step 4.1: Authentication
- Authenticates with Manheim API using client credentials
- Caches access token (expires in ~30 minutes)
- Re-authenticates automatically when token expires

#### Step 4.2: Fetch Pricing for Each Vehicle
For each vehicle:
1. **Extract VIN and Mileage**
   - Validates VIN exists (not "Not Found")
   - Converts mileage to integer (removes commas)
   
2. **Call Manheim API**
   - Endpoint: `/valuations/vin/{vin}?grade=45&odometer={mileage}`
   - Uses cached Bearer token
   - Requests wholesale average pricing
   
3. **Extract Adjusted Price**
   - Gets `adjustedPricing.wholesale.average` from response
   - Stores as `mmr_price` in vehicle dictionary
   
4. **Wait 0.5 seconds** between API calls (rate limiting)

#### Step 4.3: Output
- Returns vehicles with `mmr_price` added
- Logs successful vs failed pricing lookups

---

### **PHASE 5: LEAD QUALIFICATION** (`LeadQualifier`)

#### Step 5.1: Clean Prices
For each vehicle:
- **Original Price**: Cleans price string (removes $, commas, non-digits)
- **MMR Price**: Already numeric from API

#### Step 5.2: Calculate Price Differences
- `price_difference` = MMR Price - Original Price
- `price_difference_pct` = (difference / MMR Price) × 100

#### Step 5.3: Qualification Logic
**Qualified Lead Criteria:**
```
Original Price < MMR Price
```

**Process:**
1. Skip if either price is missing/None
2. If `original_price < mmr_price`:
   - Mark as `is_qualified_lead = True`
   - Add to `qualified_leads` list
   - Log with savings amount
3. Else:
   - Mark as `is_qualified_lead = False`
   - Add to `non_leads` list

#### Step 5.4: Output
- Returns two lists: `qualified_leads`, `non_leads`
- Logs qualification statistics

---

### **PHASE 6: DATA EXPORT** (`DataExporter`)

#### Step 6.1: Export All Vehicles
- Exports complete dataset to: `scraped_vehicles_YYYYMMDD_HHMMSS.csv`
- Includes all scraped data + phone numbers + MMR pricing + qualification status

#### Step 6.2: Export Qualified Leads
- Exports only qualified leads to: `qualified_leads_YYYYMMDD_HHMMSS.csv`
- Prioritized column order for readability

---

### **PHASE 7: SALESFORCE INTEGRATION** (`SalesforceIntegration`)

#### Step 7.1: Process Each Qualified Lead
For each qualified lead:
1. **Skip if no phone number** (can't contact anyway)
2. **Parse name from title**:
   - Attempts to extract name from vehicle title
   - Falls back to "Vehicle Owner" if not found
3. **Get primary phone** (first phone number from list)
4. **Build JSON payload**:
   ```json
   {
     "contact": {
       "first_name": "Vehicle",
       "last_name": "Owner",
       "email": "",
       "phone": "555-123-4567"
     },
     "vehicle": {
       "vin": "2C3CDZC91PH540676",
       "mileage": "2400"
     }
   }
   ```
5. **POST to Salesforce API**:
   - URL: `https://iframe-backend.vercel.app/api/submit-chatbot-form`
   - Content-Type: `application/json`
   - Timeout: 10 seconds
6. **Track results**: sent, failed, skipped
7. **Wait 0.5 seconds** between requests

#### Step 7.2: Output
- Logs Salesforce submission results
- Returns statistics: sent, failed, skipped

---

### **PHASE 8: SMS OUTREACH** (`SMSOutreach`)

#### Step 8.1: Process Each Qualified Lead
For each qualified lead:
1. **Skip if no phone number**
2. **Format phone number** to E.164 format (`+1XXXXXXXXXX`)
3. **Send SMS via Twilio**:
   - Uses initial message from config
   - Sends to primary phone number only
   - Logs message SID on success
4. **Track results**: sent, failed, no_phone
5. **Wait 0.5 seconds** between messages (rate limiting)

#### Step 8.2: Output
- Logs SMS outreach results
- Returns statistics

---

### **PHASE 9: FINAL SUMMARY**

#### Step 9.1: Generate Summary
Logs comprehensive summary:
- Total execution time (minutes)
- Total vehicles scraped
- Vehicles with phone numbers
- Qualified leads count
- Non-leads count
- File paths for exported data

#### Step 9.2: Cleanup
- Closes browser
- Finalizes log file

---

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. WEB SCRAPING (CraigslistScraper)                        │
│    → Navigate to Craigslist                                 │
│    → Apply filters (owner, bundle duplicates)              │
│    → Scrape each listing                                    │
│    → Extract: title, price, mileage, VIN, details         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PHONE EXTRACTION (PhoneExtractor)                       │
│    → Extract phone numbers from vehicle_details            │
│    → Validate and format (XXX-XXX-XXXX)                    │
│    → Add to vehicle data                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. MMR PRICING (MMRPricingService)                          │
│    → Authenticate with Manheim API                         │
│    → For each vehicle: Fetch MMR price using VIN + mileage │
│    → Cache tokens, add mmr_price to vehicle                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. LEAD QUALIFICATION (LeadQualifier)                      │
│    → Compare: Original Price vs MMR Price                   │
│    → If Original < MMR: QUALIFIED LEAD ✓                   │
│    → Calculate price differences and percentages           │
│    → Split into qualified_leads and non_leads              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. DATA EXPORT (DataExporter)                               │
│    → Export all vehicles to CSV                            │
│    → Export qualified leads to separate CSV                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. SALESFORCE INTEGRATION (SalesforceIntegration)          │
│    → For each qualified lead:                              │
│      - Build JSON payload                                   │
│      - POST to Salesforce API                              │
│      - Track success/failure                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. SMS OUTREACH (SMSOutreach)                               │
│    → For each qualified lead with phone:                   │
│      - Format phone number                                  │
│      - Send SMS via Twilio                                  │
│      - Track results                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. SUMMARY & CLEANUP                                        │
│    → Generate execution summary                             │
│    → Close browser                                          │
│    → Finalize logs                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

### Error Handling
- Try-catch blocks around all major operations
- Graceful degradation (continues on individual failures)
- Comprehensive logging of errors

### Rate Limiting
- 0.5 second delay between API calls (Manheim, Salesforce, Twilio)
- 2 second delay between page scrapes
- Prevents API throttling

### Data Persistence
- All data exported to timestamped CSV files
- Detailed logging to timestamped log file
- Preserves all intermediate results

### Modularity
- Each phase is a separate class
- Can skip phases via command-line arguments
- Easy to test individual components

---

## Command-Line Usage

### Basic Usage
```bash
python allrounder.py
```
Runs complete pipeline: scraping → extraction → pricing → qualification → Salesforce → SMS

### Skip Scraping (Use Existing Data)
```bash
python allrounder.py --skip-scraping
```
Uses existing CSV data instead of scraping

### Skip Salesforce
```bash
python allrounder.py --skip-salesforce
```
Runs everything except Salesforce submission

### Skip SMS
```bash
python allrounder.py --skip-sms
```
Runs everything except SMS outreach

### Headless Mode
```bash
python allrounder.py --headless
```
Runs browser in headless mode (no GUI)

### Custom URL
```bash
python allrounder.py --url "https://example.craigslist.org/search/cta"
```
Override target Craigslist URL

### Combined Options
```bash
python allrounder.py --skip-salesforce --skip-sms --headless
```
Scraping and qualification only, in headless mode

---

## Required Environment Variables

```env
# Manheim API
MANHEIM_ENDPOINT=https://api.manheim.com/oauth/token
MANHEIM_AUTH=base64_encoded_credentials
MANHEIM_API_BASE_URL=https://api.manheim.com/v1

# Twilio
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxx
TWILIO_PHONE_NUMBER=+1234567890

# OpenAI
OPENAI_API_KEY=sk-xxxxxxxxxxxxx

# Optional
INITIAL_MESSAGE=Custom message here
```

---

## Output Files

1. **Scraped Vehicles CSV**: `scraped_vehicles_YYYYMMDD_HHMMSS.csv`
   - All vehicles with complete data

2. **Qualified Leads CSV**: `qualified_leads_YYYYMMDD_HHMMSS.csv`
   - Only vehicles where Original Price < MMR Price

3. **Log File**: `pipeline_YYYYMMDD_HHMMSS.log`
   - Complete execution log

---

## Summary

This is a **complete end-to-end automation pipeline** that:
1. ✅ Scrapes Craigslist vehicle listings
2. ✅ Extracts phone numbers from listings
3. ✅ Fetches MMR pricing from Manheim API
4. ✅ Qualifies leads (price < MMR)
5. ✅ Sends qualified leads to Salesforce
6. ✅ Sends SMS to qualified leads via Twilio
7. ✅ Exports all data to CSV files
8. ✅ Logs everything comprehensively

The system is **production-ready** with error handling, rate limiting, logging, and modular design.

