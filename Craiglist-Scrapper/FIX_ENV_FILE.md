# How to Fix .env File Parsing Errors

## Problem

Your messages failed to send because:
1. ❌ `.env` file has parsing errors on **lines 5 and 6**
2. ❌ `TWILIO_PHONE_NUMBER` is not being loaded correctly
3. ❌ Twilio is receiving a placeholder value `+1234567890` instead of your real phone number

## Solution

### Step 1: Check Your `.env` File Format

Your `.env` file should look exactly like this (with your actual values):

```env
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890
OPENAI_API_KEY=sk-your-actual-openai-key-here
```

### Step 2: Common `.env` File Mistakes to Avoid

#### ❌ **WRONG** - Spaces around `=`
```env
TWILIO_PHONE_NUMBER = +12019756595
```

#### ✅ **CORRECT** - No spaces
```env
TWILIO_PHONE_NUMBER=+12019756595
```

#### ❌ **WRONG** - Quotes not needed for simple values
```env
TWILIO_PHONE_NUMBER="+12019756595"
```
(Quotes work but aren't necessary unless there are spaces)

#### ✅ **CORRECT** - No quotes needed
```env
TWILIO_PHONE_NUMBER=+12019756595
```

#### ❌ **WRONG** - Multi-line values without proper formatting
```env
OPENAI_SYSTEM_PROMPT=You are Carly
from Car Trackers
```

#### ✅ **CORRECT** - Multi-line values must be in quotes or single line
```env
OPENAI_SYSTEM_PROMPT="You are Carly from Car Trackers. Be friendly and professional."
```

Or single line:
```env
OPENAI_SYSTEM_PROMPT=You are Carly from Car Trackers. Be friendly and professional.
```

### Step 3: Fix Lines 5 and 6

The parsing errors are on lines 5 and 6. Common issues:

1. **Check for spaces around `=`**
   - Wrong: `VARIABLE_NAME = value`
   - Correct: `VARIABLE_NAME=value`

2. **Check for special characters that need escaping**
   - If your phone number or key has special characters, ensure proper formatting

3. **Check for empty lines or comments**
   - Comments should start with `#` on their own line or at the end
   - Don't put `# comment` on the same line as a value unless the value is quoted

### Step 4: Verify Your `.env` File

After fixing, your complete `.env` file should look like:

```env
# Twilio Configuration
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_PHONE_NUMBER=+1234567890

# OpenAI Configuration
OPENAI_API_KEY=sk-your-actual-openai-key-here

# Optional: Custom System Prompt
OPENAI_SYSTEM_PROMPT=You are Carly from Car Trackers. You are reaching out to vehicle owners to collect VIN numbers for their vehicles. Be friendly, professional, and helpful.

# Optional: Custom Initial Message
INITIAL_MESSAGE=Hello this is Carly from Car Trackers, I'm reaching out for your vehicle that you have posted on Craigslist for sale. Could you please share the VIN number of your vehicle?

# Optional: Webhook Server Configuration
WEBHOOK_PORT=5000
FLASK_DEBUG=False
```

**Important Notes:**
- Replace `+12019756595` with your **actual Twilio phone number** (format: +1XXXXXXXXXX)
- Replace the API keys with your **actual credentials**
- Make sure there are **no spaces around the `=` sign**
- Each variable should be on its **own line**

### Step 5: Test Again

After fixing the `.env` file, run:

```bash
python messaging.py
```

The script will now:
1. ✅ Show you exactly which variables are missing (if any)
2. ✅ Validate that `TWILIO_PHONE_NUMBER` is not a placeholder
3. ✅ Send messages successfully

## Quick Checklist

- [ ] `.env` file exists in the project root directory
- [ ] No spaces around `=` signs: `VAR=value` not `VAR = value`
- [ ] `TWILIO_PHONE_NUMBER` is your actual Twilio phone number (format: +1XXXXXXXXXX)
- [ ] All required variables are present:
  - [ ] `TWILIO_ACCOUNT_SID`
  - [ ] `TWILIO_AUTH_TOKEN`
  - [ ] `TWILIO_PHONE_NUMBER`
  - [ ] `OPENAI_API_KEY`
- [ ] No parsing errors (check lines 5 and 6 specifically)
- [ ] File encoding is UTF-8 (no special characters causing issues)

## Still Having Issues?

If you're still seeing parsing errors:

1. **Open your `.env` file in a text editor** (Notepad++ or VS Code, not Word)
2. **Check lines 5 and 6** specifically
3. **Remove any hidden characters** or special formatting
4. **Re-type lines 5 and 6** if needed
5. **Save the file** and try again

