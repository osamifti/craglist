# Postman ke liye cURL Commands - Craigslist Scraper

## Step 1: Server Start Karein

Pehle Flask server ko start karein:

```bash
cd Craiglist-Scrapper
python messaging.py server
```

Server `http://localhost:5001` par chalega (ya jo bhi PORT env variable me set hai).

---

## Step 2: Postman me Import Karein

### Option A: Postman me Direct Import

1. Postman kholo
2. **Import** button click karo
3. **Raw text** tab select karo
4. Neeche diye gaye curl commands ko copy-paste karo

### Option B: cURL se Direct Test

Terminal me directly bhi test kar sakte ho.

---

## Available Endpoints

### 1. Pipeline Run (Complete Scraping)

**Full Pipeline (Sab kuch run karega):**
```bash
curl -X POST http://localhost:5001/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Scraping Skip Karke (Existing CSV use karega):**
```bash
curl -X POST http://localhost:5001/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "skip_scraping": true
  }'
```

**Salesforce Skip Karke:**
```bash
curl -X POST http://localhost:5001/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "skip_salesforce": true
  }'
```

**SMS Skip Karke:**
```bash
curl -X POST http://localhost:5001/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "skip_sms": true
  }'
```

**Headless Mode (Browser GUI nahi dikhega):**
```bash
curl -X POST http://localhost:5001/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "headless": true
  }'
```

**Custom URL ke saath:**
```bash
curl -X POST http://localhost:5001/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://bakersfield.craigslist.org/search/cta"
  }'
```

**Multiple Options (Scraping + Qualification only):**
```bash
curl -X POST http://localhost:5001/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "skip_salesforce": true,
    "skip_sms": true,
    "headless": true
  }'
```

**GET Method se (Query Parameters):**
```bash
curl -X GET "http://localhost:5001/pipeline/run?skip_scraping=false&skip_salesforce=false&skip_sms=false&headless=false"
```

---

### 2. Pipeline Stop

**Running Pipeline ko Stop Karein:**
```bash
curl -X POST http://localhost:5001/pipeline/stop \
  -H "Content-Type: application/json"
```

**GET Method se:**
```bash
curl -X GET http://localhost:5001/pipeline/stop
```

---

### 3. Pipeline Status Check

**Current Status Dekho:**
```bash
curl -X GET http://localhost:5001/status
```

**Phone Number ke saath Thread ID:**
```bash
curl -X GET "http://localhost:5001/status?phone=+1234567890"
```

---

### 4. API Info

**Available Endpoints List:**
```bash
curl -X GET http://localhost:5001/api
```

---

### 5. Conversations API

**Sab Conversations:**
```bash
curl -X GET http://localhost:5001/api/conversations
```

**Specific Thread ke Messages:**
```bash
curl -X GET http://localhost:5001/api/conversations/THREAD_ID_HERE
```

**Inbound Messages:**
```bash
curl -X GET http://localhost:5001/api/conversations/THREAD_ID_HERE/inbound
```

**Outbound Messages:**
```bash
curl -X GET http://localhost:5001/api/conversations/THREAD_ID_HERE/outbound
```

---

## Postman me Setup

### Request 1: Run Full Pipeline

1. **Method:** POST
2. **URL:** `http://localhost:5001/pipeline/run`
3. **Headers:**
   - `Content-Type: application/json`
4. **Body (raw JSON):**
```json
{}
```

### Request 2: Run Pipeline (Scraping Skip)

1. **Method:** POST
2. **URL:** `http://localhost:5001/pipeline/run`
3. **Headers:**
   - `Content-Type: application/json`
4. **Body (raw JSON):**
```json
{
  "skip_scraping": true
}
```

### Request 3: Run Pipeline (Headless Mode)

1. **Method:** POST
2. **URL:** `http://localhost:5001/pipeline/run`
3. **Headers:**
   - `Content-Type: application/json`
4. **Body (raw JSON):**
```json
{
  "headless": true,
  "skip_salesforce": false,
  "skip_sms": false
}
```

### Request 4: Check Status

1. **Method:** GET
2. **URL:** `http://localhost:5001/status`

### Request 5: Stop Pipeline

1. **Method:** POST
2. **URL:** `http://localhost:5001/pipeline/stop`
3. **Headers:**
   - `Content-Type: application/json`

---

## Parameters Explanation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_scraping` | boolean | false | Agar true, to scraping skip karega aur existing CSV use karega |
| `skip_salesforce` | boolean | false | Salesforce submission skip karega |
| `skip_sms` | boolean | false | SMS sending skip karega |
| `headless` | boolean | null | Browser headless mode me chalega (GUI nahi dikhega) |
| `url` | string | null | Custom Craigslist URL override karega |

---

## Response Examples

### Success Response (Pipeline Started):
```json
{
  "success": true,
  "message": "Pipeline run started",
  "status": {
    "is_running": true,
    "started_at": "2024-01-15T10:30:00",
    "stop_requested": false
  }
}
```

### Error Response (Already Running):
```json
{
  "success": false,
  "error": "Pipeline is already running",
  "status": {
    "is_running": true,
    "started_at": "2024-01-15T10:30:00"
  }
}
```

---

## Important Notes

1. **Server pehle start karo:** `python messaging.py server`
2. **Default port:** 5001 (ya PORT env variable)
3. **Pipeline long running hai:** Scraping me time lag sakta hai
4. **Status check karo:** `/status` endpoint se current state dekho
5. **Stop karne ke liye:** `/pipeline/stop` endpoint use karo

---

## Troubleshooting

### Server Start Nahi Ho Raha?
- Check karo ki port 5001 available hai
- `.env` file me required variables set hain
- Dependencies install hain: `pip install -r requirements.txt`

### Pipeline Start Nahi Ho Raha?
- Server running hai ya nahi check karo
- `/status` endpoint se current state dekho
- Logs check karo (terminal me dikhenge)

### Response 409 (Conflict)?
- Matlab pipeline already running hai
- Pehle `/pipeline/stop` karo, phir dobara try karo


