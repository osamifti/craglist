# How to Send Messages - Complete Guide

There are several ways to send SMS messages using your bot. Choose the method that best fits your needs.

## Method 1: Send Messages from CSV (Bulk)

Send messages to all phone numbers in your `extracted_phones.csv` file.

### Command:
```bash
python messaging.py
```

This will:
- Read phone numbers from `extracted_phones.csv`
- Send the initial message (from `.env` or default) to each number
- Show progress and summary

### Example Output:
```
Starting SMS Bot...
Message: Hello this is Carly from Car Trackers...

Reading phone numbers from column: 'Extracted Phone Numbers'
Found 2 phone numbers

[1/2] Sending to +19895753748...
✓ Message sent to +19895753748 - SID: SM1234567890
[2/2] Sending to +16613048222...
✓ Message sent to +16613048222 - SID: SM0987654321

=== Summary ===
Total numbers: 2
Successfully sent: 2
Failed: 0
```

---

## Method 2: Send Single Message via Script

Use the helper script to send a message to a single phone number.

### Command:
```bash
# Using default message from .env
python send_message.py +1234567890

# With custom message
python send_message.py +1234567890 "Hello, this is a test message"
```

### Examples:
```bash
# Send default message
python send_message.py +19895753748

# Send custom message
python send_message.py +19895753748 "Hi! Could you please share your VIN number?"
```

---

## Method 3: Send via API Endpoint (HTTP)

Use the `/send` endpoint to send messages via HTTP requests.

### Using GET Request (Browser or curl)

**Browser:**
```
http://localhost:5000/send?to=+1234567890&message=Hello%20World
```

**curl:**
```bash
curl "http://localhost:5000/send?to=%2B1234567890&message=Hello%20World"
```

### Using POST Request (JSON)

**curl:**
```bash
curl -X POST http://localhost:5000/send \
  -H "Content-Type: application/json" \
  -d '{"to": "+1234567890", "message": "Hello World"}'
```

**Python:**
```python
import requests

response = requests.post('http://localhost:5000/send', json={
    'to': '+1234567890',
    'message': 'Hello World'
})
print(response.json())
```

### Using POST Request (Form Data)

**curl:**
```bash
curl -X POST http://localhost:5000/send \
  -d "to=+1234567890" \
  -d "message=Hello World"
```

### Response Format:
```json
{
  "success": true,
  "message": "Message sent successfully",
  "to": "+1234567890",
  "thread_id": "+1234567890"
}
```

### With Thread ID:
```bash
curl "http://localhost:5000/send?to=%2B1234567890&message=Hello&thread_id=conv_12345"
```

---

## Method 4: Send Directly from Python Code

You can import and use the `send_sms` function directly in your Python code.

### Example:
```python
from messaging import send_sms, format_phone_number

# Send a message
phone = "+1234567890"
message = "Hello, this is a test message"

success = send_sms(phone, message, initialize_conversation=True)

if success:
    print("Message sent successfully!")
else:
    print("Failed to send message")
```

### In a Script:
```python
#!/usr/bin/env python3
from messaging import send_sms, INITIAL_MESSAGE

# Send to multiple numbers
phone_numbers = [
    "+1234567890",
    "+0987654321"
]

for phone in phone_numbers:
    send_sms(phone, INITIAL_MESSAGE)
    print(f"Sent to {phone}")
```

---

## Phone Number Format

The bot automatically normalizes phone numbers. You can use any of these formats:

- `+1234567890` (E.164 - recommended)
- `1234567890` (10 digits)
- `(123) 456-7890` (formatted)
- `123-456-7890` (dashed)

All will be normalized to: `+11234567890`

---

## Message Content

### Using Default Message
The default message is defined in your `.env` file:
```env
INITIAL_MESSAGE=Hello this is Carly from Car Trackers, I'm reaching out for your vehicle that you have posted on Craigslist for sale. Could you please share the VIN number of your vehicle?
```

### Using Custom Message
You can specify any message when sending:
- Via script: `python send_message.py +1234567890 "Your custom message"`
- Via API: Include `message` parameter in your request

---

## Thread ID (Optional)

When sending messages, you can optionally include a thread ID for conversation tracking:

```bash
# Via API
curl "http://localhost:5000/send?to=%2B1234567890&message=Hello&thread_id=conv_12345"

# Via Python
from messaging import send_sms
send_sms("+1234567890", "Hello", initialize_conversation=True)
# Then use thread_id_mapping to track
```

---

## Testing Examples

### Test Single Message:
```bash
python send_message.py +1234567890 "Test message"
```

### Test via API (GET):
```bash
curl "http://localhost:5000/send?to=%2B1234567890&message=Test"
```

### Test via API (POST):
```bash
curl -X POST http://localhost:5000/send \
  -H "Content-Type: application/json" \
  -d '{"to": "+1234567890", "message": "Test"}'
```

### Send to All Numbers in CSV:
```bash
python messaging.py
```

---

## Troubleshooting

### Message Not Sending
1. Check your `.env` file has correct Twilio credentials
2. Verify `TWILIO_PHONE_NUMBER` is your actual Twilio number (not placeholder)
3. Check Twilio account has sufficient credits
4. Review error messages in the console

### API Endpoint Not Working
1. Make sure webhook server is running: `python messaging.py server`
2. Check server is listening on correct port (default: 5000)
3. Verify endpoint URL: `http://localhost:5000/send`

### Phone Number Format Error
- Ensure phone number is valid (10-11 digits for US numbers)
- Include country code: `+1` for US numbers
- Use format: `+1XXXXXXXXXX`

---

## Quick Reference

| Method | Command | Use Case |
|--------|---------|----------|
| CSV Bulk | `python messaging.py` | Send to many numbers from CSV |
| Single Script | `python send_message.py +1234567890 "msg"` | Quick test, single message |
| API GET | `curl "http://localhost:5000/send?to=+123&message=Hello"` | HTTP requests, testing |
| API POST | `curl -X POST http://localhost:5000/send -d "to=+123&message=Hello"` | Production, automation |
| Python Code | `from messaging import send_sms` | Integration in other scripts |

---

## Summary

**For bulk sending (CSV):** `python messaging.py`  
**For single message:** `python send_message.py +1234567890 "message"`  
**For API/automation:** Use `/send` endpoint  
**For integration:** Import `send_sms` function in your code

Choose the method that works best for your workflow!

