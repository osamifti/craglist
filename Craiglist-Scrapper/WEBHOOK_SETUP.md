# Webhook Setup Guide for SMS Bot Replies

This guide explains how to set up your bot to automatically reply to incoming messages using OpenAI.

## Overview

Your bot can now:
1. ✅ Send initial messages to phone numbers from CSV (existing functionality)
2. ✅ Receive replies from users via Twilio webhooks
3. ✅ Generate intelligent responses using OpenAI API
4. ✅ Use custom system prompt from `.env` file

## Prerequisites

- Twilio account with phone number
- OpenAI API key
- Python environment with all dependencies installed

## Step 1: Create `.env` File

Create a `.env` file in the project root with the following variables:

```env
# Twilio Configuration
TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom System Prompt
# This defines how the bot behaves when responding to users
OPENAI_SYSTEM_PROMPT=You are Carly from Car Trackers. You are reaching out to vehicle owners to collect VIN numbers for their vehicles. Be friendly, professional, and helpful. If they provide a VIN number, acknowledge it and thank them. If they ask questions about why you need the VIN, explain that you're helping with vehicle tracking and registration services. Keep your responses concise and conversational, like a text message.

# Optional: Custom Initial Message
INITIAL_MESSAGE=Hello this is Carly from Car Trackers, I'm reaching out for your vehicle. Could you please share the VIN number of your vehicle? Thank you.

# Optional: Webhook Server Configuration
WEBHOOK_PORT=5000
FLASK_DEBUG=False
```

**Important**: 
- Replace all placeholder values with your actual credentials
- Never commit your `.env` file to version control (it should be in `.gitignore`)

## Step 2: Run the Webhook Server

To enable automatic replies, you need to run the webhook server:

```bash
python messaging.py server
```

This will start a Flask server on port 5000 (or the port specified in `.env`) that listens for incoming messages from Twilio.

## Step 3: Expose Your Server to the Internet

Twilio needs to send webhooks to your server. You have two options:

### Option A: Local Development (Using ngrok)

1. Install ngrok: https://ngrok.com/download
2. In a new terminal, run:
   ```bash
   ngrok http 5000
   ```
3. Copy the HTTPS URL (e.g., `https://abc123.ngrok.io`)
4. This URL will forward requests to your local server

### Option B: Production (Deploy to a Server)

Deploy your application to a server with a public URL (e.g., Heroku, AWS, Railway, etc.)

## Step 4: Configure Twilio Webhook

1. Log in to your [Twilio Console](https://console.twilio.com/)
2. Go to **Phone Numbers** → **Manage** → **Active Numbers**
3. Click on your Twilio phone number
4. Scroll to **Messaging** section
5. Under **A MESSAGE COMES IN**, enter your webhook URL:
   - Local: `https://your-ngrok-url.ngrok.io/webhook`
   - Production: `https://your-domain.com/webhook`
6. Set method to **POST**
7. Click **Save**

## Step 5: Test the Bot

1. Make sure the webhook server is running (`python messaging.py server`)
2. Send a text message to your Twilio phone number
3. The bot should automatically reply using OpenAI

## How It Works

1. **User sends message** → Twilio receives it
2. **Twilio sends webhook** → POST request to your `/webhook` endpoint
3. **Bot processes message** → Extracts message and sender phone number
4. **OpenAI generates response** → Uses system prompt from `.env` and conversation history
5. **Bot sends reply** → TwiML response sent back to user via Twilio

## Conversation History

The bot maintains conversation history per phone number (last 10 messages) to provide context-aware responses. This history is stored in memory and will reset when the server restarts.

## Troubleshooting

### Bot not replying
- Check that webhook server is running: `python messaging.py server`
- Verify Twilio webhook URL is correctly configured
- Check server logs for errors
- Verify OpenAI API key is correct in `.env`

### OpenAI API errors
- Check your API key is valid and has credits
- Verify the key is set in `.env` file
- Check OpenAI API status: https://status.openai.com/

### Webhook not receiving messages
- Ensure your server is accessible from the internet (use ngrok for local)
- Check Twilio webhook URL is correct
- Verify Flask server logs for incoming requests

## Running Both Modes

You can run the bot in two modes:

1. **Send Messages Mode** (default):
   ```bash
   python messaging.py
   ```
   Reads CSV and sends initial messages.

2. **Webhook Server Mode**:
   ```bash
   python messaging.py server
   ```
   Starts server to receive and reply to messages.

You can run both simultaneously in different terminals if needed.

