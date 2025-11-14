# Webhook GET Method & Thread ID Support

## Overview

The webhook now supports both **GET** and **POST** methods, making it easier to test and integrate with different systems. The bot also supports **Thread ID** tracking for better conversation management.

## Features

✅ **GET Method Support** - Test webhooks easily via browser or curl  
✅ **POST Method Support** - Standard Twilio webhook format  
✅ **Thread ID Management** - Track conversations with custom thread IDs  
✅ **Phone Number Extraction** - Automatically extracts and normalizes phone numbers  
✅ **Status Endpoint** - Check server health and thread information  

## Webhook Endpoint

### URL
```
/webhook
```

### Supported Methods
- **GET** - For testing and alternative integrations
- **POST** - Standard Twilio webhook (recommended for production)

## Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `Body` | The message content from the user | `"Hello, I have a question"` |
| `From` | The sender's phone number | `"+1234567890"` or `"1234567890"` |

### Optional Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `ThreadId` | Custom thread ID for conversation tracking | `"conv_12345"` |

## Usage Examples

### 1. Testing with GET Method (Browser)

Open in your browser:
```
http://localhost:5000/webhook?Body=Hello&From=+1234567890
```

Or with Thread ID:
```
http://localhost:5000/webhook?Body=Hello&From=+1234567890&ThreadId=conv_12345
```

### 2. Testing with GET Method (curl)

```bash
curl "http://localhost:5000/webhook?Body=Hello%20world&From=%2B1234567890"
```

Or with Thread ID:
```bash
curl "http://localhost:5000/webhook?Body=Hello&From=%2B1234567890&ThreadId=conv_12345"
```

### 3. Production (POST from Twilio)

Twilio automatically sends POST requests with these parameters:
- `Body` - Message content
- `From` - Sender phone number
- Optional: `ThreadId` - If using Twilio Conversations API

### 4. Testing POST with curl

```bash
curl -X POST http://localhost:5000/webhook \
  -d "Body=Hello world" \
  -d "From=+1234567890"
```

Or with Thread ID:
```bash
curl -X POST http://localhost:5000/webhook \
  -d "Body=Hello" \
  -d "From=+1234567890" \
  -d "ThreadId=conv_12345"
```

## Thread ID System

### How Thread IDs Work

1. **Default Behavior**: If no ThreadId is provided, the phone number is used as the thread ID
2. **Custom Thread ID**: You can provide a custom ThreadId parameter to track conversations differently
3. **Thread Mapping**: The system maintains a mapping between phone numbers and thread IDs

### Thread ID Examples

**Example 1: Default (Phone Number as Thread ID)**
```
GET /webhook?Body=Hello&From=+1234567890
```
- Thread ID: `+1234567890` (same as phone number)

**Example 2: Custom Thread ID**
```
GET /webhook?Body=Hello&From=+1234567890&ThreadId=conv_abc123
```
- Thread ID: `conv_abc123`
- Phone Number: `+1234567890`
- Mapping stored: `+1234567890` → `conv_abc123`

### Getting Thread ID for a Phone Number

Use the status endpoint:
```
GET /status?phone=+1234567890
```

Response:
```json
{
  "status": "online",
  "server_time": "2025-11-03T23:00:00",
  "total_conversations": 5,
  "total_threads": 5,
  "phone": "+1234567890",
  "thread_id": "conv_abc123",
  "message_count": 10
}
```

## Status Endpoint

### URL
```
/status
```

### Method
- **GET** only

### Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `phone` | Optional | Phone number to get thread info for | `+1234567890` |

### Response Examples

**General Status (no phone parameter):**
```json
{
  "status": "online",
  "server_time": "2025-11-03T23:00:00.000000",
  "total_conversations": 5,
  "total_threads": 5
}
```

**Specific Phone Number:**
```json
{
  "status": "online",
  "server_time": "2025-11-03T23:00:00.000000",
  "total_conversations": 5,
  "total_threads": 5,
  "phone": "+1234567890",
  "thread_id": "+1234567890",
  "message_count": 10
}
```

## Phone Number Format

The bot automatically normalizes phone numbers to E.164 format:

- Input: `1234567890` → Output: `+11234567890`
- Input: `+1234567890` → Output: `+1234567890`
- Input: `(123) 456-7890` → Output: `+11234567890`

## Logging

All webhook requests are logged with:
- Request method (GET/POST)
- Thread ID
- Phone number (both original and normalized)
- Message content
- Response status

Example log output:
```
2025-11-03 23:00:00 - INFO - Webhook called via GET method
2025-11-03 23:00:00 - INFO - Received message - Thread ID: +1234567890, From: +1234567890 (+1234567890), Message: Hello
2025-11-03 23:00:01 - INFO - Sending AI-generated response - Thread ID: +1234567890, To: +1234567890, Response: Thank you for your message...
```

## Testing Workflow

1. **Start the webhook server:**
   ```bash
   python messaging.py server
   ```

2. **Test with GET (browser):**
   ```
   http://localhost:5000/webhook?Body=Test%20message&From=%2B1234567890
   ```

3. **Check status:**
   ```
   http://localhost:5000/status?phone=%2B1234567890
   ```

4. **Test with POST (curl):**
   ```bash
   curl -X POST http://localhost:5000/webhook \
     -d "Body=Hello" \
     -d "From=+1234567890"
   ```

## Production Setup

For production with Twilio:

1. **Configure Twilio Webhook URL:**
   - Go to Twilio Console → Phone Numbers → Your Number
   - Set webhook URL: `https://your-domain.com/webhook`
   - Method: **POST** (recommended)

2. **Twilio automatically sends:**
   - `Body` - Message content
   - `From` - Sender phone number
   - Other Twilio metadata

3. **The bot will:**
   - Extract phone number from `From` parameter
   - Use phone number as thread ID (unless custom ThreadId provided)
   - Generate AI response using conversation history
   - Return TwiML XML response

## Troubleshooting

### Webhook not receiving messages
- Check server is running: `python messaging.py server`
- Verify webhook URL is accessible from internet (use ngrok for local testing)
- Check logs for incoming requests

### Thread ID not working
- Verify ThreadId parameter is being sent
- Check status endpoint to see current thread mapping
- Review logs to see thread ID being used

### Phone number not recognized
- Check phone number format (should be E.164: +1XXXXXXXXXX)
- Verify `From` parameter is being sent correctly
- Check logs for normalized phone number

