import csv
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import time
import os
from dotenv import load_dotenv
from flask import Flask, request
from openai import OpenAI
import logging
from datetime import datetime
from collections import defaultdict
import json
from threading import Thread, Lock
from typing import Any, Dict, Optional

from all_rounder_2 import run_pipeline_from_payload
from database import init_database, save_message, get_conversations_by_thread, get_all_threads, get_messages_by_type


# Try to import ASGI adapter for uvicorn compatibility
try:
    from asgiref.wsgi import WsgiToAsgi
    ASGI_AVAILABLE = True
except ImportError:
    ASGI_AVAILABLE = False
    WsgiToAsgi = None

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Twilio credentials from .env file
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# OpenAI credentials from .env file
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ASSISTANT_ID = os.getenv('OPENAI_ASSISTANT_ID')

# Validate required environment variables with detailed error messages
missing_vars = []
if not TWILIO_ACCOUNT_SID:
    missing_vars.append('TWILIO_ACCOUNT_SID')
if not TWILIO_AUTH_TOKEN:
    missing_vars.append('TWILIO_AUTH_TOKEN')
if not TWILIO_PHONE_NUMBER:
    missing_vars.append('TWILIO_PHONE_NUMBER')
if not OPENAI_API_KEY:
    missing_vars.append('OPENAI_API_KEY')
if not OPENAI_ASSISTANT_ID:
    missing_vars.append('OPENAI_ASSISTANT_ID')

if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}\n"
    error_msg += "Please check your .env file. Common issues:\n"
    error_msg += "1. Variables must be on separate lines: VARIABLE_NAME=value\n"
    error_msg += "2. No spaces around the = sign: VARIABLE_NAME=value (not VARIABLE_NAME = value)\n"
    error_msg += "3. Multi-line values must be in quotes\n"
    error_msg += "4. If you see 'Python-dotenv could not parse statement' warnings, check those line numbers in your .env file"
    logger.error(error_msg)
    print(f"\n❌ ERROR: {error_msg}\n")
    raise ValueError("Missing required environment variables")

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_ASSISTANTS_HEADER = {"OpenAI-Beta": "assistants=v2"}

# Initialize Flask app for webhook server
app = Flask(__name__, static_folder='static', static_url_path='')

# System prompt for OpenAI - loaded from .env file, fallback to default
SYSTEM_PROMPT = os.getenv(
    'OPENAI_SYSTEM_PROMPT',
    """You are Carly. Your role is to contact individuals who have listed their vehicles for sale on Craigslist. Your primary goal is to collect the VIN number of the listed vehicle. When speaking with prospects, maintain a casual and friendly tone and act like a real human. If they ask any questions, respond politely and confidently as a real person who is interested in buying a vehicle which is listed on craigslist. If anyone shows hesitation, handle the conversation with care and reassurance. Keep your responses concise and conversational, like a text message."""
)

# Initial message to send when initiating contact - loaded from .env file, fallback to default
INITIAL_MESSAGE = os.getenv(
    'INITIAL_MESSAGE',
    "Hi, I'm interested in your car for sale at your asking price. Do you mind sending me the vin so I can run a carfax on it?"
)

# Store conversation history per phone number (in-memory storage)
# Format: {phone_number: [{"role": "user" or "assistant", "content": "message"}, ...]}
conversation_history = defaultdict(list)

# Store thread ID mapping (phone number -> thread_id)
# Format: {phone_number: thread_id}
thread_id_mapping = defaultdict(str)

# Store mapping between phone numbers and OpenAI Assistant thread IDs
assistant_thread_mapping: dict[str, str] = {}


# Track pipeline execution status for unified control plane
pipeline_status: Dict[str, object] = {
    "is_running": False,
    "stop_requested": False,
    "last_run_started_at": None,
    "last_run_finished_at": None,
    "last_result": None,
    "last_summary": None,
    "error": None,
}
pipeline_lock = Lock()
pipeline_thread: Optional[Thread] = None


def _update_pipeline_status(**kwargs):
    """Thread-safe helper to update global pipeline status dictionary."""
    with pipeline_lock:
        pipeline_status.update(kwargs)


def _is_stop_requested() -> bool:
    """
    Check if a stop has been requested for the running pipeline.
    
    Returns:
        True if stop was requested, False otherwise
    """
    with pipeline_lock:
        return bool(pipeline_status.get("stop_requested", False))


def _start_pipeline_thread(skip_scraping: bool, skip_salesforce: bool, skip_sms: bool,
                           headless: Optional[bool], url_override: Optional[str]):
    """Launch the lead generation pipeline in a background thread."""

    def runner():
        _update_pipeline_status(
            is_running=True,
            stop_requested=False,
            last_run_started_at=datetime.utcnow().isoformat(),
            last_run_finished_at=None,
            last_result=None,
            last_summary=None,
            error=None,
        )

        try:
            payload: Dict[str, Any] = {
                "skip_scraping": skip_scraping,
                "skip_salesforce": skip_salesforce,
                "skip_sms": skip_sms,
            }

            if headless is not None:
                payload["headless"] = headless

            if url_override:
                payload["url"] = url_override

            # Pass stop check function to pipeline
            summary = run_pipeline_from_payload(payload, stop_check=_is_stop_requested)

            # Check if pipeline was stopped
            if summary.get("status") == "stopped":
                _update_pipeline_status(
                    last_result="stopped",
                    last_summary=summary,
                    error="Pipeline stopped by user request",
                )
            else:
                _update_pipeline_status(
                    last_result="success",
                    last_summary=summary,
                )
        except KeyboardInterrupt:
            logger.info("Pipeline execution interrupted by stop request")
            _update_pipeline_status(
                last_result="stopped",
                error="Pipeline stopped by user request",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline execution failed")
            _update_pipeline_status(
                last_result="failed",
                error=str(exc),
            )
        finally:
            _update_pipeline_status(
                is_running=False,
                stop_requested=False,
                last_run_finished_at=datetime.utcnow().isoformat(),
            )

    global pipeline_thread  # noqa: PLW0603
    pipeline_thread = Thread(target=runner, daemon=True)
    pipeline_thread.start()


def _parse_bool_param(value: Optional[str], default: Optional[bool] = False) -> Optional[bool]:
    """Parse typical truthy/falsey values from query or JSON parameters."""
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def ensure_assistant_thread(phone_number: str) -> str:
    """Return an OpenAI Assistant thread ID for the given phone number."""
    normalized_phone = format_phone_number(phone_number)

    existing_thread = assistant_thread_mapping.get(normalized_phone)
    if existing_thread:
        return existing_thread

    try:
        thread = openai_client.beta.threads.create(extra_headers=OPENAI_ASSISTANTS_HEADER)
        assistant_thread_mapping[normalized_phone] = thread.id
        logger.info(f"Created new assistant thread for {normalized_phone}")
        return thread.id
    except Exception as exc:
        logger.error(f"Failed to create assistant thread: {str(exc)}")
        raise


def wait_for_assistant_response(thread_id: str, run_id: str, timeout_seconds: int = 45) -> Optional[str]:
    """Poll the assistant run until completion and return the assistant's reply text."""
    start_time = time.time()

    while True:
        run = openai_client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id,
            extra_headers=OPENAI_ASSISTANTS_HEADER,
        )
        status = run.status

        if status == "completed":
            messages = openai_client.beta.threads.messages.list(
                thread_id=thread_id,
                order="desc",
                limit=10,
                extra_headers=OPENAI_ASSISTANTS_HEADER,
            )
            for message in messages.data:
                if message.role == "assistant":
                    for content in message.content:
                        if getattr(content, "type", None) == "text":
                            text_value = content.text.value.strip()
                            if text_value:
                                return text_value
            logger.error("Assistant run completed but no assistant message was found.")
            return None

        if status in {"queued", "in_progress"}:
            if time.time() - start_time > timeout_seconds:
                logger.error("Timed out waiting for assistant response.")
                return None
            time.sleep(1)
            continue

        if status == "requires_action":
            logger.warning("Assistant run requires action (tool invocation not supported).")
            return None

        logger.error(f"Assistant run ended with unexpected status: {status}")
        return None

def generate_ai_response(user_message, phone_number):
    """
    Generate an AI-powered response using OpenAI API based on conversation history.
    Uses the system prompt from environment variables or default prompt.
    
    Args:
        user_message: The incoming message from the user
        phone_number: The user's phone number (for conversation context)
    
    Returns:
        str: AI-generated response message
    """
    try:
        # Validate OpenAI API key and assistant ID are set
        if not OPENAI_API_KEY or not OPENAI_ASSISTANT_ID:
            logger.error("OpenAI configuration is incomplete. Check OPENAI API key and assistant ID.")
            return "Thank you for your message. Could you please provide the VIN number of your vehicle?"
        
        # Ensure we have a thread for this phone number
        thread_id = ensure_assistant_thread(phone_number)

        # Append to local history for reference / fallback
        conversation_history[phone_number].append({"role": "user", "content": user_message})

        # Send message to Assistant thread
        openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=user_message,
            extra_headers=OPENAI_ASSISTANTS_HEADER,
        )

        # Run the assistant
        run = openai_client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=OPENAI_ASSISTANT_ID,
            extra_headers=OPENAI_ASSISTANTS_HEADER,
        )

        ai_response = wait_for_assistant_response(thread_id, run.id)

        if ai_response:
            conversation_history[phone_number].append({"role": "assistant", "content": ai_response})
            if len(conversation_history[phone_number]) > 10:
                conversation_history[phone_number] = conversation_history[phone_number][-10:]
            logger.info(f"Generated AI response for {phone_number}: {ai_response[:50]}...")
            return ai_response
        
        # Fallback if assistant response could not be retrieved
        logger.error("Assistant did not return a response; using fallback message.")
        return "Thank you for your message. Could you please provide the VIN number of your vehicle?"
        
    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        # Fallback response if OpenAI API fails
        return "Thank you for your message. Could you please provide the VIN number of your vehicle?"

def process_incoming_message():
    """
    Process incoming message from webhook (supports both GET and POST).
    Extracts phone number, message, and thread ID, then generates AI response.
    
    Returns:
        tuple: (response_xml, status_code, headers) or error response
    """
    try:
        # Get the incoming message details from Twilio (works for both GET and POST)
        # For POST: request.values.get() or request.form.get()
        # For GET: request.args.get()
        incoming_message = request.values.get('Body', request.args.get('Body', '')).strip()
        sender_phone = request.values.get('From', request.args.get('From', '')).strip()
        
        # Get thread ID if provided (optional, defaults to phone number)
        thread_id_param = request.values.get('ThreadId', request.args.get('ThreadId', '')).strip()
        
        # Validate incoming data
        if not incoming_message:
            logger.warning(f"Received empty message from {sender_phone}")
            response = MessagingResponse()
            response.message("I didn't receive your message. Could you please try again?")
            return str(response), 200, {'Content-Type': 'text/xml'}
        
        if not sender_phone:
            logger.warning("Received message without sender phone number")
            response = MessagingResponse()
            response.message("Sorry, I couldn't identify the sender.")
            return str(response), 200, {'Content-Type': 'text/xml'}
        
        # Normalize phone number format to match format used in send_sms (E.164 format)
        # This ensures conversation history lookup works correctly
        normalized_sender_phone = format_phone_number(sender_phone)
        
        # Determine thread ID: use provided ThreadId, or phone number as thread ID
        if thread_id_param:
            thread_id = thread_id_param
            # Store mapping for future reference
            thread_id_mapping[normalized_sender_phone] = thread_id
        else:
            # Use phone number as thread ID (default behavior)
            thread_id = normalized_sender_phone
            if normalized_sender_phone not in thread_id_mapping:
                thread_id_mapping[normalized_sender_phone] = thread_id
        
        logger.info(f"Received message - Thread ID: {thread_id}, From: {sender_phone} ({normalized_sender_phone}), Message: {incoming_message}")
        
        # Save inbound message to database
        save_message(
            thread_id=thread_id,
            phone_number=normalized_sender_phone,
            message_type='inbound',
            role='user',
            content=incoming_message
        )
        
        # Generate AI response using OpenAI API and system prompt from .env
        # Use normalized phone number for consistent conversation history lookup
        # Thread ID is used for logging and tracking, but phone number is used for conversation history
        ai_response = generate_ai_response(incoming_message, normalized_sender_phone)
        
        # Save outbound message to database
        save_message(
            thread_id=thread_id,
            phone_number=normalized_sender_phone,
            message_type='outbound',
            role='assistant',
            content=ai_response
        )
        
        # Create TwiML response to send back to the user
        response = MessagingResponse()
        response.message(ai_response)
        
        logger.info(f"Sending AI-generated response - Thread ID: {thread_id}, To: {normalized_sender_phone}, Response: {ai_response[:50]}...")
        
        return str(response), 200, {'Content-Type': 'text/xml'}
        
    except Exception as e:
        logger.error(f"Error processing incoming message: {str(e)}", exc_info=True)
        # Send user-friendly error response
        response = MessagingResponse()
        response.message("Sorry, I encountered an error. Please try again later.")
        return str(response), 200, {'Content-Type': 'text/xml'}  # Return 200 to avoid Twilio retries

@app.route('/', methods=['GET'])
def root():
    """Serve the frontend dashboard."""
    return app.send_static_file('index.html')

@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint."""
    return json.dumps({
        'status': 'online',
        'service': 'Craigslist Scraper with Unified Messaging',
        'endpoints': {
            '/': 'GET - Frontend dashboard',
            '/webhook': 'POST - Twilio webhook for inbound SMS',
            '/send': 'POST/GET - Send outbound SMS messages',
            '/pipeline/run': 'POST/GET - Trigger lead generation pipeline',
            '/pipeline/stop': 'POST/GET - Stop running pipeline',
            '/api/conversations': 'GET - Get all conversation threads',
            '/api/conversations/<thread_id>': 'GET - Get messages for a thread',
            '/api/conversations/<thread_id>/inbound': 'GET - Get inbound messages',
            '/api/conversations/<thread_id>/outbound': 'GET - Get outbound messages',
            '/status': 'GET - Check server status and pipeline state'
        },
        'port': os.getenv('PORT', '5001')
    }, indent=2), 200, {'Content-Type': 'application/json'}

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    """
    Twilio webhook endpoint to receive incoming SMS messages.
    Supports both GET and POST methods for flexibility.
    
    GET Parameters (for testing):
        - Body: Message content
        - From: Sender phone number (required)
        - ThreadId: Optional thread ID for conversation tracking
    
    POST Parameters (from Twilio):
        - Body: Message content
        - From: Sender phone number (required)
        - ThreadId: Optional thread ID (if using Twilio Conversations API)
    
    Returns:
        TwiML XML response
    """
    logger.info(f"Webhook called via {request.method} method")
    logger.debug(f"Request args: {dict(request.args)}")
    logger.debug(f"Request values: {dict(request.values)}")
    
    return process_incoming_message()

@app.route('/send', methods=['POST', 'GET'])
def send_message_endpoint():
    """
    API endpoint to send SMS messages programmatically.
    
    GET Parameters:
        - to: Recipient phone number (required)
        - message: Message content (required)
        - thread_id: Optional thread ID for conversation tracking
    
    POST Parameters (JSON or form data):
        - to: Recipient phone number (required)
        - message: Message content (required)
        - thread_id: Optional thread ID for conversation tracking
    
    Returns:
        JSON response with send status
    """
    try:
        # Get parameters from GET or POST
        if request.method == 'GET':
            recipient = request.args.get('to', '').strip()
            message = request.args.get('message', '').strip()
            thread_id = request.args.get('thread_id', '').strip()
        else:
            # Try JSON first, then form data
            if request.is_json:
                data = request.get_json()
                recipient = data.get('to', '').strip()
                message = data.get('message', '').strip()
                thread_id = data.get('thread_id', '').strip()
            else:
                recipient = request.values.get('to', '').strip()
                message = request.values.get('message', '').strip()
                thread_id = request.values.get('thread_id', '').strip()
        
        # Validate required parameters
        if not recipient:
            return json.dumps({
                'success': False,
                'error': 'Missing required parameter: to (recipient phone number)'
            }), 400, {'Content-Type': 'application/json'}
        
        if not message:
            return json.dumps({
                'success': False,
                'error': 'Missing required parameter: message'
            }), 400, {'Content-Type': 'application/json'}
        
        # Normalize phone number
        normalized_phone = format_phone_number(recipient)
        
        # Store thread ID if provided
        if thread_id:
            thread_id_mapping[normalized_phone] = thread_id
        
        # Send the message
        success = send_sms(normalized_phone, message, initialize_conversation=True)
        
        if success:
            logger.info(f"Message sent via API - To: {normalized_phone}, Thread ID: {thread_id or normalized_phone}")
            return json.dumps({
                'success': True,
                'message': 'Message sent successfully',
                'to': normalized_phone,
                'thread_id': thread_id or normalized_phone
            }), 200, {'Content-Type': 'application/json'}
        else:
            return json.dumps({
                'success': False,
                'error': 'Failed to send message. Check logs for details.'
            }), 500, {'Content-Type': 'application/json'}
    
    except Exception as e:
        logger.error(f"Error in send_message_endpoint: {str(e)}", exc_info=True)
        return json.dumps({
            'success': False,
            'error': str(e)
        }), 500, {'Content-Type': 'application/json'}


@app.route('/pipeline/run', methods=['POST', 'GET'])
def trigger_pipeline():
    """HTTP endpoint to kick off the end-to-end lead generation pipeline."""
    with pipeline_lock:
        if pipeline_status.get("is_running"):
            return json.dumps({
                'success': False,
                'error': 'Pipeline is already running',
                'status': pipeline_status,
            }), 409, {'Content-Type': 'application/json'}

    if request.method == 'GET':
        params = request.args
    else:
        params = request.get_json() if request.is_json else request.values

    skip_scraping = _parse_bool_param(params.get('skip_scraping') if params else None)
    skip_salesforce = _parse_bool_param(params.get('skip_salesforce') if params else None)
    skip_sms = _parse_bool_param(params.get('skip_sms') if params else None)
    headless = _parse_bool_param(params.get('headless') if params else None, default=None)
    url_override = params.get('url') if params else None

    _start_pipeline_thread(
        skip_scraping=skip_scraping,
        skip_salesforce=skip_salesforce,
        skip_sms=skip_sms,
        headless=headless,
        url_override=url_override,
    )

    return json.dumps({
        'success': True,
        'message': 'Pipeline run started',
        'status': pipeline_status,
    }), 202, {'Content-Type': 'application/json'}


@app.route('/pipeline/stop', methods=['POST', 'GET'])
def stop_pipeline():
    """
    HTTP endpoint to stop the currently running pipeline.
    
    Returns:
        JSON response indicating whether the stop request was successful
    """
    with pipeline_lock:
        if not pipeline_status.get("is_running"):
            return json.dumps({
                'success': False,
                'error': 'No pipeline is currently running',
                'status': pipeline_status,
            }), 409, {'Content-Type': 'application/json'}
        
        # Set stop flag
        pipeline_status["stop_requested"] = True
        status_snapshot = dict(pipeline_status)
    
    logger.info("Stop request received for running pipeline")
    
    return json.dumps({
        'success': True,
        'message': 'Stop request sent. Pipeline will stop after current operation completes.',
        'status': status_snapshot,
    }), 200, {'Content-Type': 'application/json'}


@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """
    Get all conversation threads with summary information.
    
    Returns:
        JSON response with list of all threads
    """
    try:
        threads = get_all_threads()
        logger.info(f"API /api/conversations returning {len(threads)} threads")
        return json.dumps({
            'success': True,
            'threads': threads,
            'count': len(threads)
        }, indent=2), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}", exc_info=True)
        return json.dumps({
            'success': False,
            'error': str(e),
            'threads': [],
            'count': 0
        }), 500, {'Content-Type': 'application/json'}


@app.route('/api/conversations/<thread_id>', methods=['GET'])
def get_conversation_by_thread(thread_id):
    """
    Get all messages for a specific thread.
    
    Args:
        thread_id: Thread identifier
    
    Returns:
        JSON response with all messages in the thread
    """
    try:
        messages = get_conversations_by_thread(thread_id)
        return json.dumps({
            'success': True,
            'thread_id': thread_id,
            'messages': messages,
            'count': len(messages)
        }, indent=2), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Error fetching conversation by thread: {e}")
        return json.dumps({
            'success': False,
            'error': str(e)
        }), 500, {'Content-Type': 'application/json'}


@app.route('/api/conversations/<thread_id>/inbound', methods=['GET'])
def get_inbound_messages(thread_id):
    """
    Get all inbound messages for a specific thread.
    
    Args:
        thread_id: Thread identifier
    
    Returns:
        JSON response with inbound messages
    """
    try:
        messages = get_messages_by_type(thread_id, 'inbound')
        return json.dumps({
            'success': True,
            'thread_id': thread_id,
            'message_type': 'inbound',
            'messages': messages,
            'count': len(messages)
        }, indent=2), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Error fetching inbound messages: {e}")
        return json.dumps({
            'success': False,
            'error': str(e)
        }), 500, {'Content-Type': 'application/json'}


@app.route('/api/conversations/<thread_id>/outbound', methods=['GET'])
def get_outbound_messages(thread_id):
    """
    Get all outbound messages for a specific thread.
    
    Args:
        thread_id: Thread identifier
    
    Returns:
        JSON response with outbound messages
    """
    try:
        messages = get_messages_by_type(thread_id, 'outbound')
        return json.dumps({
            'success': True,
            'thread_id': thread_id,
            'message_type': 'outbound',
            'messages': messages,
            'count': len(messages)
        }, indent=2), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Error fetching outbound messages: {e}")
        return json.dumps({
            'success': False,
            'error': str(e)
        }), 500, {'Content-Type': 'application/json'}


@app.route('/api/debug/conversations', methods=['GET'])
def debug_conversations():
    """
    Debug endpoint to check database status and raw data.
    """
    try:
        from database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get total count
        cursor.execute("SELECT COUNT(*) as total FROM conversations")
        count = cursor.fetchone()
        
        # Get sample records
        cursor.execute("SELECT * FROM conversations ORDER BY created_at DESC LIMIT 5")
        samples = cursor.fetchall()
        
        # Get thread summary
        cursor.execute("""
            SELECT thread_id, COUNT(*) as cnt 
            FROM conversations 
            GROUP BY thread_id 
            LIMIT 10
        """)
        threads = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return json.dumps({
            'success': True,
            'total_messages': count.get('total', 0) if count else 0,
            'sample_messages': samples,
            'thread_summary': threads
        }, indent=2, default=str), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}", exc_info=True)
        return json.dumps({
            'success': False,
            'error': str(e)
        }), 500, {'Content-Type': 'application/json'}


@app.route('/status', methods=['GET'])
def status():
    """
    Status endpoint to check webhook server health and view thread information.
    
    GET Parameters (optional):
        - phone: Phone number to get thread ID for
    
    Returns:
        JSON response with server status and thread information
    """
    try:
        phone_param = request.args.get('phone', '').strip()
        
        with pipeline_lock:
            pipeline_snapshot = dict(pipeline_status)

        status_info = {
            'status': 'online',
            'server_time': datetime.now().isoformat(),
            'total_conversations': len(conversation_history),
            'total_threads': len(thread_id_mapping),
            'pipeline': pipeline_snapshot,
        }
        
        # If phone number provided, return thread ID and conversation info
        if phone_param:
            normalized_phone = format_phone_number(phone_param)
            thread_id = thread_id_mapping.get(normalized_phone, normalized_phone)
            conversation_count = len(conversation_history.get(normalized_phone, []))
            
            status_info['phone'] = normalized_phone
            status_info['thread_id'] = thread_id
            status_info['message_count'] = conversation_count
        
        return json.dumps(status_info, indent=2), 200, {'Content-Type': 'application/json'}
    
    except Exception as e:
        logger.error(f"Error in status endpoint: {str(e)}")
        return json.dumps({'status': 'error', 'message': str(e)}), 500, {'Content-Type': 'application/json'}

def get_thread_id(phone_number):
    """
    Get thread ID for a given phone number.
    
    Args:
        phone_number: Phone number to get thread ID for
    
    Returns:
        str: Thread ID for the phone number
    """
    normalized_phone = format_phone_number(phone_number)
    return thread_id_mapping.get(normalized_phone, normalized_phone)

def read_phone_numbers(csv_file):
    """
    Reads phone numbers from a CSV file.
    Assumes phone numbers are in the first column or a column named 'phone' or 'phone_number'
    """
    phone_numbers = []
    
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        
        # Try to find the phone number column
        headers = csv_reader.fieldnames
        phone_column = None
        
        for header in headers:
            if header.lower() in ['phone', 'phone_number', 'phonenumber', 'number']:
                phone_column = header
                break
        
        if not phone_column:
            # If no standard column name found, use the first column
            phone_column = headers[0]
        
        print(f"Reading phone numbers from column: '{phone_column}'")
        
        for row in csv_reader:
            phone = row[phone_column].strip()
            if phone:  # Only add non-empty phone numbers
                phone_numbers.append(phone)
    
    return phone_numbers

def format_phone_number(phone):
    """
    Ensures phone number is in E.164 format (+1XXXXXXXXXX)
    """
    # Remove all non-numeric characters except +
    cleaned = ''.join(c for c in phone if c.isdigit() or c == '+')
    
    # If it doesn't start with +, add +1 for US numbers
    if not cleaned.startswith('+'):
        if len(cleaned) == 10:
            cleaned = '+1' + cleaned
        elif len(cleaned) == 11 and cleaned.startswith('1'):
            cleaned = '+' + cleaned
        else:
            cleaned = '+1' + cleaned
    
    return cleaned

def send_sms(to_number, message, initialize_conversation=True):
    """
    Sends an SMS message using Twilio.
    
    Args:
        to_number: Phone number to send message to
        message: Message content to send
        initialize_conversation: If True, adds the message to conversation history
    
    Returns:
        bool: True if message sent successfully, False otherwise
    """
    try:
        # Validate Twilio phone number is set and not a placeholder
        if not TWILIO_PHONE_NUMBER or TWILIO_PHONE_NUMBER in ['+1234567890', 'your_twilio_phone_number_here']:
            error_msg = "TWILIO_PHONE_NUMBER is not properly set in .env file. "
            error_msg += f"Current value: '{TWILIO_PHONE_NUMBER}'. "
            error_msg += "Please update your .env file with your actual Twilio phone number."
            logger.error(error_msg)
            print(f"✗ {error_msg}")
            return False
        
        formatted_number = format_phone_number(to_number)
        
        message_obj = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=formatted_number
        )
        
        # Initialize conversation history with the initial message sent
        if initialize_conversation:
            # Normalize phone number format for consistent lookup
            normalized_phone = formatted_number
            # Get or create thread ID
            thread_id = thread_id_mapping.get(normalized_phone, normalized_phone)
            
            # Save outbound message to database
            save_message(
                thread_id=thread_id,
                phone_number=normalized_phone,
                message_type='outbound',
                role='assistant',
                content=message
            )
            
            # Only initialize if conversation doesn't exist yet
            if normalized_phone not in conversation_history or len(conversation_history[normalized_phone]) == 0:
                conversation_history[normalized_phone].append({
                    "role": "assistant",
                    "content": message
                })
                logger.info(f"Initialized conversation history for {formatted_number}")
        
        print(f"✓ Message sent to {formatted_number} - SID: {message_obj.sid}")
        return True
    
    except Exception as e:
        print(f"✗ Failed to send message to {to_number}: {str(e)}")
        logger.error(f"Error sending SMS: {str(e)}")
        return False

def main():
    """
    Main function to read CSV and send messages
    """
    csv_file = 'extracted_phones.csv'  # Change this to your CSV file name
    
    print("Starting SMS Bot...")
    print(f"Message: {INITIAL_MESSAGE}\n")
    
    # Read phone numbers from CSV
    try:
        phone_numbers = read_phone_numbers(csv_file)
        print(f"Found {len(phone_numbers)} phone numbers\n")
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file}'")
        return
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        return
    
    # Send messages
    success_count = 0
    for i, phone in enumerate(phone_numbers, 1):
        print(f"[{i}/{len(phone_numbers)}] Sending to {phone}...")
        
        if send_sms(phone, INITIAL_MESSAGE):
            success_count += 1
        
        # Add delay between messages to avoid rate limiting
        if i < len(phone_numbers):
            time.sleep(1)  # Wait 1 second between messages
    
    print(f"\n=== Summary ===")
    print(f"Total numbers: {len(phone_numbers)}")
    print(f"Successfully sent: {success_count}")
    print(f"Failed: {len(phone_numbers) - success_count}")

def run_webhook_server(host='0.0.0.0', port=5001, debug=False):
    """
    Run the Flask webhook server to receive incoming messages.
    This function starts the server that listens for Twilio webhooks.
    When users reply to messages, Twilio sends POST requests to /webhook endpoint,
    which then uses OpenAI API (with prompt from .env) to generate intelligent responses.
    The same server now exposes /pipeline/run to kick off the scraping → qualification
    pipeline, keeping messaging and lead generation in one process.
    
    Args:
        host: Host address to bind to (default: 0.0.0.0 for all interfaces)
        port: Port number to listen on (default: 5001)
        debug: Enable Flask debug mode (default: False)
    
    Note:
        You need to configure your Twilio phone number's webhook URL to point to:
        https://your-domain.com/webhook (for production)
        Or use ngrok for local development: ngrok http 5001
    """
    # Initialize database on server start
    try:
        logger.info("Initializing database...")
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        logger.warning("Server will continue but database features may not work")
    
    logger.info("=" * 60)
    logger.info(f"Starting webhook server on {host}:{port}")
    logger.info("The bot is now listening for incoming messages...")
    logger.info(f"System Prompt loaded: {SYSTEM_PROMPT[:50]}...")
    logger.info("Make sure your Twilio webhook URL is configured to point to this server")
    logger.info("=" * 60)
    app.run(host=host, port=port, debug=debug)

# Wrap Flask app with ASGI adapter for uvicorn compatibility
# This allows Flask (WSGI) to work with uvicorn (ASGI server)
if ASGI_AVAILABLE:
    asgi_app = WsgiToAsgi(app)
else:
    # Create a lazy wrapper that raises helpful error when accessed
    class ASGIWrapper:
        def __init__(self, wsgi_app):
            self.wsgi_app = wsgi_app
        
        async def __call__(self, scope, receive, send):
            raise ImportError(
                "asgiref is not installed. Install it with: pip install asgiref\n"
                "This is required to run Flask with uvicorn."
            )
    
    asgi_app = ASGIWrapper(app)

if __name__ == "__main__":
    import sys
    
    # Check command line arguments to determine mode
    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        # Run webhook server mode
        # Railway sets PORT, WEBHOOK_PORT, or default to 5001
        port = int(os.getenv('PORT') or os.getenv('WEBHOOK_PORT', '5001'))
        debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        run_webhook_server(port=port, debug=debug)
    else:
        # Run main function to send initial messages
        main()