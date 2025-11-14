"""
Simple script to send a single SMS message.
Usage: python send_message.py <phone_number> <message>
Example: python send_message.py +1234567890 "Hello, this is a test message"
"""

import sys
import os
from messaging import send_sms, format_phone_number, INITIAL_MESSAGE

def send_single_message(phone_number, message=None):
    """
    Send a single SMS message to a phone number.
    
    Args:
        phone_number: Recipient phone number
        message: Message content (uses INITIAL_MESSAGE if not provided)
    """
    if message is None:
        message = INITIAL_MESSAGE
    
    print(f"Sending message to {phone_number}...")
    print(f"Message: {message}\n")
    
    normalized_phone = format_phone_number(phone_number)
    success = send_sms(normalized_phone, message, initialize_conversation=True)
    
    if success:
        print(f"✓ Message sent successfully to {normalized_phone}")
    else:
        print(f"✗ Failed to send message to {normalized_phone}")
    
    return success

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python send_message.py <phone_number> [message]")
        print("\nExamples:")
        print('  python send_message.py +1234567890')
        print('  python send_message.py +1234567890 "Hello, this is a test"')
        sys.exit(1)
    
    phone = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else None
    
    send_single_message(phone, message)

