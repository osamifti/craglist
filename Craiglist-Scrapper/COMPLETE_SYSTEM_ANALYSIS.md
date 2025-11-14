# Complete System Analysis - Craigslist Scraper & SMS Bot

This document provides a comprehensive analysis of all Python files in the system and how they work together.

---

## 📋 System Overview

This is a **3-stage automation pipeline** for scraping Craigslist vehicle listings and automating SMS outreach:

```
Stage 1: Web Scraping (scraper.py)
    ↓
Stage 2: Phone Extraction (number-extractor.py)
    ↓
Stage 3: SMS Automation (messaging.py + send_message.py)
```

---

## 📁 File 1: `scraper.py` - Web Scraping Module

### Purpose
Automated web scraper that extracts vehicle listings from Bakersfield Craigslist using Selenium WebDriver.

### Key Components

#### 1. **BakersfieldCraigslistScraper Class**

**Initialization (`__init__`):**
- Sets up Chrome WebDriver (headless or visible mode)
- Configures target URL: Bakersfield Craigslist vehicle listings
- Defines XPath/CSS selectors for all page elements
- Creates timestamped CSV file for output
- Sets up logging system

**Key Attributes:**
- `target_url`: Specific Craigslist search URL with filters
- `driver`: Selenium WebDriver instance
- `wait`: WebDriverWait for explicit waits
- `csv_filename`: Timestamped output file (e.g., `bakersfield_craigslist_detailed_20251103_233636.csv`)

#### 2. **Core Methods**

**`_setup_chrome_driver()`:**
- Configures Chrome browser options (headless, window size, user agent)
- Automatically downloads ChromeDriver using webdriver-manager
- Creates WebDriver instance with 15-second timeout

**`_setup_csv_file()`:**
- Creates CSV file with headers: `Title, Price, Mileage, VIN, Vehicle Details, Scraped At`

**`navigate_to_target_url()`:**
- Loads target URL in browser
- Waits 40 seconds for page to fully load (dynamic content)

**`click_owner_filter()`:**
- Finds and clicks "owner" filter button using XPath
- Scrolls to element before clicking for visibility

**`click_bundle_duplicates()`:**
- Finds and clicks "bundleDuplicates" checkbox
- Ensures duplicates are bundled together

**`get_gallery_cards()`:**
- Finds all vehicle listing cards on the page
- Returns list of WebElement objects

**`click_gallery_card()`:**
- Clicks individual vehicle card to open details page
- Uses JavaScript click as fallback if regular click fails
- Waits for detail page to load

**`click_show_contact()`:**
- Clicks "show-contact" button **twice** (5 seconds apart)
- Reveals contact information hidden by Craigslist
- Handles cases where contact is already visible

**`scrape_vehicle_details()`:**
- Extracts all vehicle information:
  - Title (from `#titletextonly`)
  - Price (from `.price` class)
  - Mileage (from `.attr.auto_miles`)
  - VIN (from `.attr.auto_vin`)
  - Vehicle Details (from `#postingbody`)
- Returns dictionary with all scraped data
- Handles missing elements gracefully (returns "Not Found")

**`go_back_to_gallery()`:**
- Navigates back to gallery page
- Handles double-back navigation if contact modal was opened
- Waits for gallery cards to reload

**`run_scraping_process()` - Main Orchestration:**
1. Navigate to target URL (wait 40 seconds)
2. Click owner filter button
3. Click bundleDuplicates checkbox
4. Wait 5 seconds
5. Get all gallery cards
6. For each card:
   - Click card to open details
   - Click show-contact button (twice)
   - Scrape vehicle details
   - Save to CSV
   - Go back to gallery
   - Move to next card
7. Close browser

**`_append_to_csv()`:**
- Appends scraped data to CSV file
- Includes timestamp for each record

### Data Flow
```
Browser → Navigate URL → Apply Filters → Find Cards → 
For Each Card: Open → Scrape → Save CSV → Back → Next
```

### Output
- CSV file: `bakersfield_craigslist_detailed_YYYYMMDD_HHMMSS.csv`
- Columns: Title, Price, Mileage, VIN, Vehicle Details, Scraped At

### Usage
```bash
python scraper.py
# Prompts for headless mode (y/n)
# Runs complete scraping process
```

---

## 📁 File 2: `number-extractor.py` - Phone Number Extraction

### Purpose
Extracts phone numbers from the scraped vehicle details CSV file using regex pattern matching.

### Key Components

#### 1. **`extract_phone_numbers(text: str)`**

**Functionality:**
- Uses multiple regex patterns to find US phone numbers
- Supports formats:
  - `(555) 555-5555`
  - `555-555-5555`
  - `555.555.5555`
  - `555 555 5555`
  - `5555555555`

**Validation:**
- Validates area code starts with 2-9 (US phone standard)
- Validates exchange code starts with 2-9
- Prevents false positives (dates, IDs, etc.)

**Normalization:**
- Formats all numbers as: `XXX-XXX-XXXX`
- Removes duplicates while preserving order
- Only returns valid 10-digit numbers

**Regex Patterns:**
```python
r'\b\(?([2-9]\d{2})\)?[-.\s]?([2-9]\d{2})[-.\s]?(\d{4})\b'
r'\b([2-9]\d{2})[-.\s]?([2-9]\d{2})[-.\s]?(\d{4})\b'
```

#### 2. **`process_csv(input_file, output_file)`**

**Functionality:**
- Reads CSV file from scraper output
- Finds "Vehicle Details" column (case-insensitive)
- Extracts phone numbers from each row's vehicle details text
- Prints found numbers to console
- Returns list of dictionaries with extracted data

**Output Structure:**
```python
{
    'original_data': 'Full vehicle details text',
    'extracted_phones': ['555-123-4567', '555-987-6543'],
    'phone_count': 2,
    # ... other CSV columns
}
```

#### 3. **`write_results_to_csv(results, output_file)`**

**Functionality:**
- Writes extracted phone numbers to new CSV
- Single column: "Extracted Phone Numbers"
- Multiple numbers per row joined with commas
- Empty cells for rows with no phone numbers

### Data Flow
```
Input CSV (from scraper.py) → Read "Vehicle Details" column → 
Extract phone numbers → Output to extracted_phones.csv
```

### Usage
```bash
python number-extractor.py
# Reads: bakersfield_craigslist_detailed_YYYYMMDD_HHMMSS.csv
# Outputs: extracted_phones.csv
```

---

## 📁 File 3: `messaging.py` - SMS Bot & Webhook Server

### Purpose
Complete SMS automation system with:
- Bulk SMS sending from CSV
- Webhook server for receiving replies
- OpenAI-powered intelligent responses
- Conversation history management
- Thread ID tracking

### Key Components

#### 1. **Initialization & Configuration**

**Environment Variables (from `.env`):**
- `TWILIO_ACCOUNT_SID`: Twilio account identifier
- `TWILIO_AUTH_TOKEN`: Twilio authentication token
- `TWILIO_PHONE_NUMBER`: Your Twilio phone number
- `OPENAI_API_KEY`: OpenAI API key for AI responses
- `OPENAI_SYSTEM_PROMPT`: Custom bot personality/instructions
- `INITIAL_MESSAGE`: Default message to send
- `WEBHOOK_PORT`: Server port (default: 5000)

**Initialization:**
- Loads environment variables using `python-dotenv`
- Validates all required credentials are present
- Initializes Twilio Client
- Initializes OpenAI Client
- Creates Flask app for webhook server
- Sets up logging system

#### 2. **Data Structures**

**`conversation_history`:**
- Type: `defaultdict(list)`
- Format: `{phone_number: [{"role": "user/assistant", "content": "message"}, ...]}`
- Purpose: Stores last 10 messages per phone number for context
- Location: In-memory (lost on server restart)

**`thread_id_mapping`:**
- Type: `defaultdict(str)`
- Format: `{phone_number: thread_id}`
- Purpose: Maps phone numbers to custom thread IDs
- Default: Phone number itself if no thread ID provided

#### 3. **Core Functions**

**`generate_ai_response(user_message, phone_number)`:**
- **Purpose**: Generate intelligent reply using OpenAI API
- **Process**:
  1. Get conversation history for phone number
  2. Build messages array: [system prompt, history, current message]
  3. Call OpenAI API (gpt-3.5-turbo, 200 tokens, temp 0.7)
  4. Extract AI response
  5. Update conversation history (add user message + AI response)
  6. Limit history to last 10 messages
- **Returns**: AI-generated response string
- **Fallback**: Default message if OpenAI fails

**`format_phone_number(phone)`:**
- **Purpose**: Normalize phone numbers to E.164 format
- **Input**: Any format (1234567890, (123) 456-7890, +1234567890)
- **Output**: `+1XXXXXXXXXX` format
- **Logic**: Removes non-digits except +, adds +1 if missing

**`send_sms(to_number, message, initialize_conversation=True)`:**
- **Purpose**: Send SMS via Twilio API
- **Process**:
  1. Validate Twilio phone number is set (not placeholder)
  2. Format recipient phone number
  3. Create Twilio message via API
  4. Initialize conversation history if first message
  5. Return success/failure
- **Logging**: Logs message SID and status

**`read_phone_numbers(csv_file)`:**
- **Purpose**: Read phone numbers from CSV file
- **Logic**:
  - Auto-detects phone column (checks: 'phone', 'phone_number', 'phonenumber', 'number')
  - Falls back to first column if no match
  - Returns list of phone number strings

#### 4. **Flask Webhook Endpoints**

**`/webhook` (GET, POST):**
- **Purpose**: Receive incoming SMS messages from Twilio
- **Parameters**:
  - `Body`: Message content (required)
  - `From`: Sender phone number (required)
  - `ThreadId`: Optional thread ID
- **Process**:
  1. Extract message and sender phone
  2. Normalize phone number
  3. Determine thread ID (provided or use phone number)
  4. Generate AI response using conversation history
  5. Return TwiML XML response to Twilio
- **Response**: TwiML XML with bot's reply

**`/send` (GET, POST):**
- **Purpose**: API endpoint to send SMS messages programmatically
- **Parameters**:
  - `to`: Recipient phone number (required)
  - `message`: Message content (required)
  - `thread_id`: Optional thread ID
- **Process**:
  1. Extract parameters (GET query params or POST JSON/form data)
  2. Validate required parameters
  3. Normalize phone number
  4. Store thread ID if provided
  5. Send SMS via Twilio
  6. Return JSON response
- **Response**: JSON with success status and details

**`/status` (GET):**
- **Purpose**: Check server health and view thread information
- **Parameters**:
  - `phone`: Optional phone number to get info for
- **Response**: JSON with server status, conversation counts, thread info

#### 5. **Main Functions**

**`main()`:**
- **Purpose**: Bulk SMS sending from CSV
- **Process**:
  1. Read phone numbers from `extracted_phones.csv`
  2. Send initial message to each number
  3. Wait 1 second between messages (rate limiting)
  4. Print summary statistics
- **Usage**: `python messaging.py` (default mode)

**`run_webhook_server(host, port, debug)`:**
- **Purpose**: Start Flask webhook server
- **Process**:
  1. Log startup information
  2. Start Flask development server
  3. Listen for incoming webhook requests
- **Usage**: `python messaging.py server`

### Execution Modes

**Mode 1: Bulk SMS Sending**
```bash
python messaging.py
```
- Reads CSV and sends messages to all numbers

**Mode 2: Webhook Server**
```bash
python messaging.py server
```
- Starts Flask server on port 5000
- Listens for incoming messages
- Generates AI responses automatically

### Data Flow

**Sending Messages:**
```
CSV File → Read Numbers → Format Phone → Twilio API → SMS Sent
```

**Receiving Messages:**
```
User SMS → Twilio → POST /webhook → Extract Data → 
Generate AI Response → TwiML Response → Twilio → User
```

### Conversation Flow Example

1. **Bot sends**: "Hello, could you share your VIN?"
   - History: `[{role: "assistant", content: "Hello, could you share your VIN?"}]`

2. **User replies**: "Sure, it's 1HGBH41JXMN109186"
   - History retrieved: Previous assistant message
   - Sent to OpenAI: System prompt + history + new user message
   - AI generates contextual response
   - History updated: `[assistant msg, user msg, new assistant response]`

3. **User replies again**: "Is that all you need?"
   - Bot remembers entire conversation context
   - Responds appropriately

---

## 📁 File 4: `send_message.py` - Single Message Helper

### Purpose
Simple command-line script to send a single SMS message for testing or manual use.

### Key Components

**`send_single_message(phone_number, message=None)`:**
- **Purpose**: Send one SMS message
- **Parameters**:
  - `phone_number`: Recipient phone number
  - `message`: Optional custom message (uses INITIAL_MESSAGE if not provided)
- **Process**:
  1. Format phone number
  2. Send via `send_sms()` function from messaging.py
  3. Print success/failure status

### Usage
```bash
# With default message
python send_message.py +1234567890

# With custom message
python send_message.py +1234567890 "Hello, this is a test"
```

### Data Flow
```
Command Line Args → Format Phone → Send SMS → Print Result
```

---

## 🔄 Complete System Workflow

### Stage 1: Web Scraping
```bash
python scraper.py
```
1. Opens Chrome browser
2. Navigates to Bakersfield Craigslist
3. Applies filters (owner, bundle duplicates)
4. Scrapes each vehicle listing
5. Saves to: `bakersfield_craigslist_detailed_YYYYMMDD_HHMMSS.csv`

**Output:** CSV with vehicle details including phone numbers in text

### Stage 2: Phone Extraction
```bash
python number-extractor.py
```
1. Reads scraped CSV file
2. Extracts phone numbers from "Vehicle Details" column
3. Validates and formats phone numbers
4. Saves to: `extracted_phones.csv`

**Output:** CSV with extracted phone numbers

### Stage 3: SMS Automation

**Option A: Bulk Send**
```bash
python messaging.py
```
1. Reads `extracted_phones.csv`
2. Sends initial message to each number
3. Initializes conversation history

**Option B: Webhook Server**
```bash
python messaging.py server
```
1. Starts Flask server on port 5000
2. Listens for incoming SMS replies
3. Generates AI responses using OpenAI
4. Maintains conversation history
5. Sends replies back via Twilio

**Option C: Single Message**
```bash
python send_message.py +1234567890 "Message"
```
1. Sends one message to specified number
2. Quick testing/manual use

---

## 🔗 Data Flow Diagram

```
┌─────────────────────┐
│   scraper.py         │
│  Web Scraping        │
│  - Selenium          │
│  - Craigslist        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  bakersfield_       │
│  craigslist_        │
│  detailed_*.csv     │
│  (Vehicle Details)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ number-extractor.py  │
│  Phone Extraction    │
│  - Regex patterns    │
│  - Validation        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ extracted_phones.csv │
│  (Phone Numbers)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   messaging.py       │
│  SMS Automation      │
│  - Twilio API        │
│  - OpenAI API        │
│  - Flask Webhook     │
└─────────────────────┘
```

---

## 📊 Key Features & Technologies

### Technologies Used
- **Selenium**: Web automation and scraping
- **Flask**: Web framework for webhook server
- **Twilio**: SMS sending and receiving
- **OpenAI API**: AI-powered message generation
- **Python-dotenv**: Environment variable management
- **Regex**: Phone number pattern matching

### Key Features
✅ **Automated Web Scraping**: Selenium-based browser automation  
✅ **Phone Number Extraction**: Regex-based extraction with validation  
✅ **Bulk SMS Sending**: Send to multiple numbers from CSV  
✅ **Webhook Server**: Receive and respond to incoming messages  
✅ **AI-Powered Responses**: OpenAI GPT-3.5 for intelligent replies  
✅ **Conversation Memory**: Maintains context (last 10 messages)  
✅ **Thread ID Tracking**: Custom thread management  
✅ **Phone Number Normalization**: Automatic E.164 format conversion  
✅ **API Endpoints**: HTTP API for programmatic access  
✅ **Error Handling**: Comprehensive error handling throughout  

---

## 🎯 Usage Examples

### Complete Workflow
```bash
# Step 1: Scrape Craigslist
python scraper.py

# Step 2: Extract phone numbers
python number-extractor.py

# Step 3a: Send bulk messages
python messaging.py

# Step 3b: Start webhook server (in separate terminal)
python messaging.py server
```

### Testing Individual Components
```bash
# Test single message
python send_message.py +1234567890 "Test message"

# Test webhook (GET)
curl "http://localhost:5000/webhook?Body=Hello&From=+1234567890"

# Test send API
curl "http://localhost:5000/send?to=+1234567890&message=Hello"

# Check status
curl "http://localhost:5000/status?phone=+1234567890"
```

---

## 📝 Summary

This system is a **complete automation pipeline** for:
1. **Scraping** vehicle listings from Craigslist
2. **Extracting** phone numbers from scraped data
3. **Sending** automated SMS messages
4. **Receiving** and **responding** to replies with AI

All components work together seamlessly, with data flowing from web scraping → phone extraction → SMS automation, creating a fully automated outreach and response system.

