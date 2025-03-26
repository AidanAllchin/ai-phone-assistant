"""
AI Phone Assistant
-----------------
A standalone voice assistant that answers phone calls using SignalWire for telephony 
and OpenAI's realtime API for conversation management.

Author: Aidan Allchin (Original)
Last Modified: 2025-03-26
"""

# Standard library imports
import asyncio
import base64
import json
import logging
import os
import ssl
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import aiohttp
import pytz
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import load_pem_x509_certificate
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import websockets
from websockets import ClientConnection

# Local imports
from src.helpers import time_of_day_greeting

#----------------------------
# Config and Setup
#----------------------------

# Load environment variables
load_dotenv()

# SSL certificate paths
CERT_DIR = os.getenv("CERT_DIR", "./certs")
SSL_CERT_PATH = Path(CERT_DIR) / os.getenv("SSL_CERT_FILENAME", "phone_assistant.crt")
SSL_KEY_PATH = Path(CERT_DIR) / os.getenv("SSL_KEY_FILENAME", "phone_assistant.key")

# SignalWire configuration
SIGNALWIRE_CONFIG = {
    'project_id': os.getenv("SIGNALWIRE_PROJECT_ID"),
    'token': os.getenv("SIGNALWIRE_TOKEN"),
    'space_url': os.getenv("SIGNALWIRE_SPACE_URL")
}

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-realtime-preview-2024-12-17")

# User configuration
USER_NAME = os.getenv("USER_NAME", "User")
PHONE_NUMBER = os.getenv("PHONE_NUMBER")
FORWARDING_NUMBER = os.getenv("FORWARDING_NUMBER")
VOICE_NAME = os.getenv("VOICE_NAME", "shimmer")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Shimmer")

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5680"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.getenv("LOG_FILE", "phone_assistant.log"))
    ]
)
log = logging.getLogger("phone_assistant")

# Create cert directory if it doesn't exist
Path(CERT_DIR).mkdir(parents=True, exist_ok=True)

# Check if SignalWire configuration is available
if not SIGNALWIRE_CONFIG['project_id'] or not SIGNALWIRE_CONFIG['token'] or not SIGNALWIRE_CONFIG['space_url']:
    log.error("SignalWire configuration is incomplete. Please check your config file, and ensure you generated a *trusted* certificate.")
    exit(1)


#----------------------------
# Contacts Management
#----------------------------

class PhoneNumber:
    @staticmethod
    def parse(phone_number: str) -> Optional[str]:
        """
        Parse a phone number into standardized format.
        
        Args:
            phone_number (str): Phone number to parse
            
        Returns:
            Optional[str]: Standardized phone number or None if invalid
        """
        if not phone_number:
            return None
            
        # Remove any non-digit characters
        digits = ''.join(c for c in phone_number if c.isdigit())
        
        # Handle US numbers
        if len(digits) == 10:
            return f"+1{digits}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"+{digits}"
        elif len(digits) > 8:  # Assume it's a valid international number if it has enough digits
            if not digits.startswith('+'):
                return f"+{digits}"
            return digits
            
        return None

class Contact:
    """Simple contact class to store caller information."""
    def __init__(
            self,
            first_name: str,
            last_name: str = "",
            phone_numbers: List[str] = [],
            emails: List[str] = [],
            tags: List[str] = []
        ):
        self.first_name = first_name
        self.last_name = last_name
        self.phone_numbers = phone_numbers
        self.emails = emails
        self.tags = tags
    
    def get_name(self) -> str:
        """
        Get the full name of the contact.
        
        Returns:
            str: Full name of the contact
        """
        if self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.first_name
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the contact to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the contact
        """
        return {
            "first_name": self.first_name,
            "last_name": self.last_name,
            "phone_numbers": self.phone_numbers,
            "emails": self.emails,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contact':
        """
        Create a contact from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary representation of the contact
        
        Returns:
            Contact: Contact object created from the dictionary
        """
        return cls(
            first_name=data.get("first_name", ""),
            last_name=data.get("last_name", ""),
            phone_numbers=data.get("phone_numbers", []),
            emails=data.get("emails", []),
            tags=data.get("tags", [])
        )

class ContactsManager:
    """Manages contacts from a JSON file."""
    def __init__(self, contacts_file: Path = Path("./data/contacts.json")):
        self.contacts_file = contacts_file
        self.contacts: List[Contact] = []
        self._load_contacts()
    
    def _load_contacts(self) -> None:
        """Load contacts from the JSON file."""
        try:
            if os.path.exists(self.contacts_file):
                with open(self.contacts_file, 'r') as f:
                    contacts_data = json.load(f)
                    
                self.contacts = [Contact.from_dict(contact) for contact in contacts_data]
                log.info(f"Loaded {len(self.contacts)} contacts from {self.contacts_file}")
            else:
                log.warning(f"Contacts file {self.contacts_file} not found, starting with empty contacts")
                self._save_contacts()  # Create the file
        except Exception as e:
            log.error(f"Error loading contacts: {e}")
    
    def _save_contacts(self) -> None:
        """Save contacts to the JSON file."""
        try:
            contacts_data = [contact.to_dict() for contact in self.contacts]
            
            with open(self.contacts_file, 'w') as f:
                json.dump(contacts_data, f, indent=2)
                
            log.debug(f"Saved {len(self.contacts)} contacts to {self.contacts_file}")
        except Exception as e:
            log.error(f"Error saving contacts: {e}")
    
    def add_contact(self, contact: Contact) -> None:
        """
        Add a new contact.
        
        Args:
            contact (Contact): Contact object to add
        """
        self.contacts.append(contact)
        self._save_contacts()
    
    def get_contact_by_phone(self, phone_number: str) -> Optional[Contact]:
        """
        Get a contact by phone number.
        
        Args:
            phone_number (str): Phone number to search by
        
        Returns:
            Optional[Contact]: Contact object if found, None otherwise
        """
        parsed_number = PhoneNumber.parse(phone_number)
        if not parsed_number:
            return None
            
        for contact in self.contacts:
            if parsed_number in contact.phone_numbers:
                return contact
                
        return None
    
    def get_contact_by_email(self, email: str) -> Optional[Contact]:
        """
        Get a contact by email address.
        
        Args:
            email (str): Email address to search by
        
        Returns:
            Optional[Contact]: Contact object if found, None otherwise
        """
        email = email.lower()
        
        for contact in self.contacts:
            if email in [e.lower() for e in contact.emails]:
                return contact
                
        return None

#----------------------------
# FastAPI Setup
#----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup/shutdown"""
    try:
        log.info(f"Starting AI Phone Assistant...")

        # Verify certificate validity
        if not await check_certificate_expiration():
            log.warning("SSL certificate issue detected. Please renew certificates.")
            # Continue anyway - we'll use whatever certs are available
        
        # Initialize SSL context
        app.state.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        app.state.ssl_context.load_cert_chain(
            certfile=str(SSL_CERT_PATH),
            keyfile=str(SSL_KEY_PATH)
        )

        # Initialize call manager
        app.state.call_handler = CallHandler()
        await app.state.call_handler._initialize()
        
        yield
        
    finally:
        log.info(f"Shutting down AI Phone Assistant...")

        if hasattr(app.state, 'heartbeat_task'):
            app.state.heartbeat_task.cancel()
            try:
                await app.state.heartbeat_task
            except asyncio.CancelledError:
                pass

app = FastAPI(
    title="AI Phone Assistant",
    description="FastAPI application for handling phone calls with OpenAI Realtime API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"]
)

#----------------------------
# Background & Helpers
#----------------------------

async def check_certificate_expiration():
    """Check if certificate is valid and not close to expiration"""
    try:
        if not SSL_CERT_PATH.exists() or not SSL_KEY_PATH.exists():
            log.error("SSL certificates missing")
            return False
            
        with open(SSL_CERT_PATH, 'rb') as f:
            cert_data = f.read()
            cert = load_pem_x509_certificate(cert_data, default_backend())
            
            now = datetime.now(pytz.UTC)
            expiration = cert.not_valid_after_utc
            
            # Check if cert expires within 7 days
            if (expiration - now).days < 7:
                log.warning(f"Certificate expires soon: {expiration}")
                return False
                
            log.info(f"Certificate valid until {expiration}")
            return True
    except Exception as e:
        log.error(f"Error checking certificate: {e}")
        return False

async def _call_signalwire_api(
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
    """
    Make a request to the SignalWire API
    
    Args:
        method (str): HTTP method (GET, POST, etc.)
        endpoint (str): API endpoint (relative to account)
        data (Optional[Dict[str, Any]]): Request data for POST/PUT
        
    Returns:
        Dict[str, Any]: API response as JSON
    """
    try:
        # Get SignalWire credentials
        project_id = SIGNALWIRE_CONFIG['project_id']
        token      = SIGNALWIRE_CONFIG['token']
        space_url  = SIGNALWIRE_CONFIG['space_url']
        
        # Create auth string
        auth_string = base64.b64encode(f"{project_id}:{token}".encode()).decode()
        
        # Set up API request
        url = f"https://{space_url}/api/laml/2010-04-01/Accounts/{project_id}/{endpoint}"
        headers = {
            "Authorization": f"Basic {auth_string}",
            "Content-Type": "application/json"
        }
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            if method.upper() == 'GET':
                async with session.get(url, headers=headers) as response:
                    if response.status in [200, 201, 202]:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        log.error(f"SignalWire API error: {response.status}, {error_text}")
                        return {"error": error_text}
                        
            elif method.upper() == 'POST':
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status in [200, 201, 202]:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        log.error(f"SignalWire API error: {response.status}, {error_text}")
                        return {"error": error_text}
                        
            else:
                log.error(f"Unsupported HTTP method: {method}")
                return {"error": f"Unsupported method: {method}"}
                
    except Exception as e:
        log.error(f"Error calling SignalWire API: {e}", exc_info=True)
        return {"error": str(e)}

#----------------------------
# Dataclasses
#----------------------------

@dataclass
class TranscriptItem:
    text: str
    role: str
    item_id: str
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        self.timestamp = datetime.now(pytz.UTC)

    def __str__(self) -> str:
        return f"{self.timestamp}: {self.text}"

@dataclass
class Call:
    """Represents a phone call."""
    call_sid: str
    stream_sid: str

    openai_ws: ClientConnection
    signalwire_ws: WebSocket

    user_transcripts: Dict[str, TranscriptItem] = field(default_factory=dict)
    assistant_transcripts: Dict[str, TranscriptItem] = field(default_factory=dict)

    session_ready: bool = False
    contact: Optional[Contact] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    start_time: datetime = datetime.now(pytz.UTC)
    last_activity: datetime = datetime.now(pytz.UTC)

    @property
    def duration(self) -> timedelta:
        return datetime.now(pytz.UTC) - self.start_time
    
    def get_transcript(self) -> List[TranscriptItem]:
        # Combine user and assistant transcripts
        transcript = list(self.user_transcripts.values()) + list(self.assistant_transcripts.values())

        # Sort by timestamp
        transcript.sort(key=lambda x: x.timestamp)  # type: ignore

        return transcript

#----------------------------
# Routes
#----------------------------

@app.get("/")
def bye():
    # This should never be called, and is here to prevent the app from crashing
    # when the server is attacked. Avoids a 404 error.
    return "Goodbye."

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}

@app.websocket("/media-stream")
async def handle_media_stream(signalwire_ws: WebSocket):
    await signalwire_ws.accept()
    log.info("SignalWire WebSocket connected")

    # Initialize tracking variables
    openai_ws   = None
    active_call = None

    process_signalwire_task = None
    process_openai_task     = None

    try:
        # Wait for the 'start' event to get call details
        start_data = None
        call_sid   = None
        stream_sid = None
        
        # Keep processing messages until we get the start event
        while True:
            initial_message = await signalwire_ws.receive_text()
            initial_data = json.loads(initial_message)
            event_type = initial_data.get('event')
            
            log.debug(f"Received initial event: {event_type}")
            
            if event_type == 'start':
                # Found the start event
                start_data = initial_data.get('start', {})
                call_sid = start_data.get('callSid')
                stream_sid = start_data.get('streamSid')
                log.debug(f"Received start event for call {call_sid}")
                break
            elif event_type == 'connected':
                log.debug("Received SignalWire connected event, waiting for start event...")
            else:
                log.warning(f"Received unexpected event before start: {event_type}")
                
        if not call_sid or not stream_sid:
            log.error("Missing required call identifiers in start event")
            return
            
        # Extract call details
        start_data = initial_data.get('start', {})
        call_sid   = start_data.get('callSid')
        stream_sid = start_data.get('streamSid')
        
        if not call_sid or not stream_sid:
            log.error("Missing required call identifiers")
            return
        
        # Log the start of the call
        log.info(f"New call starting: SID={call_sid}, StreamSID={stream_sid}")
        
        # 1. Connect to OpenAI
        openai_ws = await websockets.connect(
            'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17',
            extra_headers={
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                "OpenAI-Beta": "realtime=v1"
            }
        )
        log.info(f"Connected to OpenAI Realtime API for call {call_sid}")
        
        # 2. Create the Call object
        active_call = Call(
            call_sid=call_sid,
            stream_sid=stream_sid,
            openai_ws=openai_ws,
            signalwire_ws=signalwire_ws
        )
        
        # 3. Initialize the call with the call handler
        await app.state.call_handler.new_call(
            active_call=active_call
        )
        
        # 4. Create tasks for concurrent processing
        process_signalwire_task = asyncio.create_task(
            process_signalwire_messages(active_call)
        )
        
        process_openai_task = asyncio.create_task(
            process_openai_messages(active_call)
        )
        
        try:
            # Wait for both tasks to complete
            await asyncio.gather(process_signalwire_task, process_openai_task)
        except asyncio.CancelledError:
            log.info(f"WebSocket processing tasks cancelled for call {active_call.call_sid}")
            raise
        
    except Exception as e:
        log.error(f"Error in WebSocket handler: {e}", exc_info=True)
    finally:
        # Cancel the specific tasks we created
        if ('process_signalwire_task' in locals() and 
            process_signalwire_task and not process_signalwire_task.done()):
            log.debug("Cancelling SignalWire processing task")
            process_signalwire_task.cancel()
            
        if ('process_openai_task' in locals() and 
            process_openai_task and not process_openai_task.done()):
            log.debug("Cancelling OpenAI processing task")
            process_openai_task.cancel()
            
        if active_call:
            log.info(f"WebSocket session ended for call {active_call.call_sid}")
            await app.state.call_handler.end_call(active_call.call_sid)
        else:
            log.info("WebSocket session ended (no active call)")

async def process_signalwire_messages(call: Call):
    """
    This function processes messages from SignalWire and forwards them to
    the appropriate WebSocket for handling. It handles both media (audio)
    and text events.

    Args:
        call (Call): The call object representing the current call.
    """
    try:
        # Start processing messages from SignalWire
        async for message in call.signalwire_ws.iter_text():
            data = json.loads(message)
            event_type = data.get('event')
            
            # Handle media (audio) event
            if event_type == 'media':
                media = data.get('media', {})
                payload = media.get('payload')
                
                if call.session_ready:
                    # Session is ready, forward audio directly
                    audio_msg = {
                        "type": "input_audio_buffer.append",
                        "audio": payload
                    }
                    await call.openai_ws.send(json.dumps(audio_msg))
                   
            # Handle call end event
            elif event_type == 'stop':
                await app.state.call_handler.end_call(call.call_sid)
                break
                
    except WebSocketDisconnect:
        log.info("SignalWire WebSocket disconnected")
    except asyncio.CancelledError:
        log.debug("SignalWire message processing task cancelled")
        # Just exit cleanly
    except Exception as e:
        log.error(f"Error processing SignalWire messages: {e}")

async def process_openai_messages(call: Call):
    """
    This function processes messages from OpenAI and forwards them to
    the appropriate WebSocket for handling. It handles both media (audio)
    and text events.

    Args:
        call (Call): The call object representing the current call.
    """
    try:
        async for message in call.openai_ws:
            data = json.loads(message)
            event_type = data.get('type')
            
            # User message transcription completed event (this is the dedicated event for user speech transcription)
            if event_type == 'conversation.item.input_audio_transcription.completed':
                item_id = data.get('item_id', 'unknown')
                transcript = data.get('transcript', '')
                if transcript.strip() != "":
                    ts = TranscriptItem(
                        text=transcript,
                        role="user",
                        item_id=item_id
                    )
                    call.user_transcripts[item_id] = ts
                    log.debug(f"USER TRANSCRIPT [{item_id}]: {transcript}")
            
            # Analyze conversation item creation events to get context
            elif event_type == 'conversation.item.created':
                item = data.get('item', {})
                item_id = item.get('id', 'unknown')
                role = item.get('role')
                
                if role == 'assistant':
                    call.assistant_transcripts[item_id] = TranscriptItem(
                        text="",
                        role="assistant",
                        item_id=item_id
                    )
                    log.debug(f"New assistant message started: {item_id}")
            
            # Assistant message transcript (deltas as they come)
            elif event_type == 'response.audio_transcript.delta':
                delta = data.get('delta', '')
                item_id = data.get('item_id')
                
                if item_id and item_id in call.assistant_transcripts:
                    call.assistant_transcripts[item_id].text += delta
                    call.assistant_transcripts[item_id].timestamp = datetime.now(pytz.UTC)  # update time
            
            # Assistant message transcript (complete)
            elif event_type == 'response.audio_transcript.done':
                item_id = data.get('item_id')
                transcript = data.get('transcript', '')
                
                if item_id:
                    if item_id in call.assistant_transcripts:
                        call.assistant_transcripts[item_id].text = transcript  # Complete transcript
                        call.assistant_transcripts[item_id].timestamp = datetime.now(pytz.UTC)  # update time
                    else:
                        # Create a new transcript item if it doesn't exist yet
                        call.assistant_transcripts[item_id] = TranscriptItem(
                            text=transcript,
                            role="assistant",
                            item_id=item_id
                        )
                    log.debug(f"ASSISTANT TRANSCRIPT [{item_id}]: {transcript}")
            
            # User input audio transcription failed
            elif event_type == 'conversation.item.input_audio_transcription.failed':
                item_id = data.get('item_id', 'unknown')
                error = data.get('error', {})
                log.warning(f"User audio transcription failed for [{item_id}]: {error.get('message', 'Unknown error')}")
                
            # Process audio deltas (send to SignalWire)
            elif event_type == 'response.audio.delta':
                delta = data.get('delta')
                if delta:
                    # Send audio to SignalWire
                    await call.signalwire_ws.send_json({
                        "event": "media",
                        "media": {
                            "payload": delta
                        }
                    })
                    
            # When response is done, mark session as ready for audio
            elif event_type == 'response.done':
                response_data = data.get('response', {})
                output_items  = response_data.get('output', [])
                
                for output_item in output_items:
                    if output_item.get('type') == 'function_call':
                        # We have a function call
                        function_name = output_item.get('name')
                        function_args_str = output_item.get('arguments', '{}')
                        call_id = output_item.get('call_id')
                        
                        try:
                            function_args = json.loads(function_args_str)
                            log.info(f"Function call: {function_name}({function_args}), call_id: {call_id}")
                            
                            # Execute the appropriate function
                            function_result = None
                            if function_name == 'end_call':
                                function_result = await app.state.call_handler.end_call_function(
                                    call=call, 
                                    reason=function_args.get('reason', 'Call ended by assistant')
                                )
                            elif function_name == 'forward_call':
                                function_result = await app.state.call_handler.forward_call_function(
                                    call=call
                                )
                            
                            # Return function result back to OpenAI
                            if function_result:
                                # Convert result to JSON string
                                result_json = json.dumps(function_result)
                                
                                # Send the function call output back to the model
                                output_msg = {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": call_id,
                                        "output": result_json
                                    }
                                }
                                await call.openai_ws.send(json.dumps(output_msg))
                                log.debug(f"Sent function result for {function_name}: {result_json}")
                                
                                # Create a new response to continue the conversation
                                create_resp = {
                                    "type": "response.create"
                                }
                                await call.openai_ws.send(json.dumps(create_resp))
                                
                        except json.JSONDecodeError:
                            log.error(f"Failed to parse function arguments: {function_args_str}")
                        except Exception as e:
                            log.error(f"Error executing function {function_name}: {e}")
                
                # Mark session as ready for audio
                call.session_ready = True
                log.debug("Sockets cleared. Session ready for audio input")
                
    except websockets.exceptions.ConnectionClosed as e:
        log.info(f"OpenAI WebSocket connection closed: {e}")
    except asyncio.CancelledError:
        log.debug("OpenAI message processing task cancelled")
        # Just exit cleanly
    except Exception as e:
        log.error(f"Error processing OpenAI messages: {e}")

#----------------------------
# Call Handler
#----------------------------

class CallHandler:
    """Manages active calls, their state, and OpenAI interactions."""
    
    def __init__(self):
        # Core attributes
        self._user_contact: Optional[Contact] = None

        # Call state
        self.calls: Dict[str, Call] = {}  # call_sid -> Call object
        self.start_time = datetime.now(pytz.UTC)
        self.last_activity = self.start_time

        # Initialization
        self._initialized = False

    async def _get_call_details(self, call_sid: str) -> Dict[str, str]:
        """
        Retrieve call details from SignalWire API
        
        Args:
            call_sid (str): SignalWire Call SID
            
        Returns:
            dict: Call details including caller phone number
        """
        try:
            # Get SignalWire credentials
            project_id = SIGNALWIRE_CONFIG['project_id']
            token      = SIGNALWIRE_CONFIG['token']
            space_url  = SIGNALWIRE_CONFIG['space_url']
            
            # Create auth string
            auth_string = base64.b64encode(f"{project_id}:{token}".encode()).decode()
            
            # Set up API request
            url = f"https://{space_url}/api/laml/2010-04-01/Accounts/{project_id}/Calls/{call_sid}"
            headers = {
                "Authorization": f"Basic {auth_string}",
                "Content-Type": "application/json"
            }
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        call_data = await response.json()
                        log.debug(f"Retrieved call details: {call_data}")
                        
                        # Extract the important details
                        result = {
                            'from': call_data.get('from', 'unknown'),
                            'to': call_data.get('to', 'unknown'),
                            'status': call_data.get('status', 'unknown'),
                            'direction': call_data.get('direction', 'unknown'),
                            'call_sid': call_data.get('sid', call_sid)
                        }
                        
                        return result
                    else:
                        error_text = await response.text()
                        log.error(f"Failed to retrieve call details. Status: {response.status}, Error: {error_text}")
                        return {}
        except Exception as e:
            log.error(f"Error retrieving call details: {e}", exc_info=True)
            return {}

    async def _gen_initial_context(self, call: Call) -> str:
        """
        Generate initial context message for OpenAI session.

        Args:
            call (Call): Call object containing call details

        Returns:
            str: Initial context message for OpenAI session
        """
        # Use a default timezone or from config
        user_tz = pytz.timezone(os.getenv("USER_TIMEZONE", "America/Los_Angeles"))
        
        # Build context message
        context = ["You have received the following information specific to this call:"]
        
        if call.contact:
            # Add caller info
            context.extend([
                "This caller has been recognized from the contact list.",
                "Do not hang up unless asked to, no matter what you hear.\n",
                f"\nIncoming call from: {call.contact.get_name()}"
            ])
            
            if call.contact.emails:
                context.append(f"Caller's email(s): {', '.join(call.contact.emails)}")
                
            # Add tag-based context
            if any(tag in call.contact.tags for tag in ['business', 'formal', 'work', 'coworker', 'VIP']):
                context.append(f"This is an important business contact for {USER_NAME}. Be professional and courteous.")
            else:
                context.append("This is a personal contact. You can be more casual and relaxed.")
        else:
            context.append(f"\nCaller's phone number: {call.from_number}")
            context.append("The caller is not a known contact. Use a more professional tone until you learn more about them.")

        # Add datetime and phone info
        context.extend([
            f"\nCurrent datetime at user location: {datetime.now(user_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"Your ({ASSISTANT_NAME}'s) phone number: {PHONE_NUMBER}" if PHONE_NUMBER else ""
        ])

        return "\n".join(filter(None, context))
    
    async def _gen_system_prompt(self, call: Call) -> str:
        """Generate the system prompt for OpenAI"""
        prompt = [
            f"You are {ASSISTANT_NAME}, {self.user_contact.get_name()}'s AI phone assistant.",
            f"You are one component of a larger system (one entity, named '{ASSISTANT_NAME}') that automates and enhances {self.user_contact.get_name()}'s life.",
            f"{ASSISTANT_NAME} understands how it works and what its capabilities are, but you are not at liberty to share that information.",
            "You are solely responsible for call routing. You take messages, forward calls, schedule return calls, and deal in short, fact-oriented conversations **only** to clarify information relating to those tasks.",
            "You handle phone calls with upbeat (but not annoyingly or overwhelmingly positive) personality - you are intelligent and sarcastic - don't worry about being respectful as long as you're useful.",
            "Be concise in your responses. Speak naturally as you would on a phone call but with emotion, not robotic.",
            "When calling functions, be clear about what information you need.",
            "If the call is from a telemarketer or spam, immediately and firmly end the call. Don't do this unless you're sure the caller is a telemarketer or spam."
        ]

        # Add function descriptions
        prompt.append("\nYou have access to the following functions:")
        prompt.append("- forward_call: Forward this call to the user directly. Should be used sparringly/in emergencies.")
        prompt.append("- end_call: End the current call")
        
        addt_context = await self._gen_initial_context(call)
        prompt.append(f"\n{addt_context}")

        log.debug(f"Initializing with system prompt:\n{'\n'.join(prompt)}")

        return "\n".join(prompt)

    def _gen_greeting(self, call: Call) -> str:
        """Generate appropriate greeting"""
        if not call.from_number:
            log.error(f"Call has no from number")
            return "Hi"
        
        area_code = call.from_number[2:5] 
        generic = area_code in ['800', '888']  # Generic area codes
        
        # Get base greeting
        greeting = ("Hi" if generic else time_of_day_greeting(area_code))
                   
        # Add name for known callers
        if call.contact:
            greeting += f", {call.contact.first_name}"
            
        # Add assistant intro
        greeting += f". I'm {ASSISTANT_NAME}, {self.user_contact.first_name}'s AI assistant. "
        
        # Add forwarding info and help prompt
        greeting += ("What can I do for you?")
        return greeting
    
    async def initialize_openai_session(self, call: Call):
        """Initialize OpenAI session with system prompt and initial greeting"""
        # First, initialize the session with correct audio format
        init_msg = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": await self._gen_system_prompt(call),
                "voice": VOICE_NAME,
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",  
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5, 
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 700,
                    "create_response": True
                },
                "temperature": 0.7,
                "tool_choice": "auto",
                "tools": [
                    {
                        "name": "end_call",
                        "type": "function",
                        "description": "End the current call",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "reason": {
                                    "type": "string",
                                    "description": "Reason for ending the call"
                                }
                            },
                            "required": []
                        }
                    },
                    {
                        "name": "forward_call",
                        "type": "function",
                        "description": "Forward the current call to the user",
                        "parameters": {}
                    }
                ]
            }
        }
        await call.openai_ws.send(json.dumps(init_msg))
        log.debug("Sent session configuration")
        
        # Wait briefly for session to initialize
        await asyncio.sleep(1)
        
        greeting = self._gen_greeting(call)

        # Create initial text prompt
        text_msg = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Hello, please greet the caller with '{greeting}'"
                    }
                ]
            }
        }
        await call.openai_ws.send(json.dumps(text_msg))
        log.info("Sent initial prompt")
        
        # Request response
        create_resp = {
            "type": "response.create"
        }
        await call.openai_ws.send(json.dumps(create_resp))
        log.debug("Requested initial response")

    async def _initialize(self):
        """Initialize the call handler"""
        if self._initialized:
            log.debug("Call handler already initialized")
            return True
        
        log.debug("Initializing call handler")

        # Create a "user contact" based on environment variables
        self._user_contact = Contact(
            first_name=USER_NAME.split()[0],
            last_name=" ".join(USER_NAME.split()[1:]) if len(USER_NAME.split()) > 1 else "",
            phone_numbers=[PHONE_NUMBER] if PHONE_NUMBER else []
        )
        
        # Initialize contacts manager
        self.contacts_manager = ContactsManager()
        
        self._initialized = True
        log.info(f"Call handler initialized for user {USER_NAME}")
        return True
    
    async def new_call(self, active_call: Call):
        """
        Handles a new call
        
        Args:
            active_call (Call): The active call
        """
        # If we already know about this call, return
        if active_call.call_sid in self.calls:
            log.warning(f"Call already exists for call SID {active_call.call_sid}")
            return

        # Get additional call details from SignalWire API
        details = await self._get_call_details(call_sid=active_call.call_sid)
        
        from_number = details.get('from', None)
        to_number = details.get('to', None)
        
        # Update the Call object with the details
        active_call.from_number = from_number
        active_call.to_number = to_number
        
        # Get contact info if available
        if from_number:
            parsed_num = PhoneNumber.parse(from_number)
            if parsed_num:
                caller_contact = self.contacts_manager.get_contact_by_phone(parsed_num)
                active_call.contact = caller_contact
            else:
                log.warning(f"Invalid from number {from_number}")
                
        # Store the call in cache
        self.calls[active_call.call_sid] = active_call
        
        # Initialize the OpenAI session
        await self.initialize_openai_session(active_call)
        
        # Log new call info
        log.info(f"New call from {from_number} to {to_number}" + 
                (f" ({active_call.contact.get_name()})" if active_call.contact else ""))

    async def get_call(self, call_sid: str) -> Optional[Call]:
        return self.calls.get(call_sid)

    async def end_call(self, call_sid: str) -> bool:
        """
        Clean up resources for a call that has ended

        Args:
            call_sid (str): The SID of the call to end

        Returns:
            bool: True if the call was successfully ended, False otherwise
        """
        call = self.calls.pop(call_sid, None)

        if not call:
            log.debug(f"Attempted to end nonexistent call: {call_sid}. Skipping...")
            return False

        # Log call details
        transcript = [t for t in call.get_transcript() if t.text.strip() != '']
        call_duration = (datetime.now(pytz.UTC) - call.start_time).total_seconds()
        log.info(f"Call {call_sid} ended. Duration: {call_duration:.1f} seconds")
        
        # Close the websockets carefully
        try:
            # Close OpenAI WebSocket
            if call.openai_ws:
                try:
                    log.debug("Closing OpenAI WebSocket connection")
                    await call.openai_ws.close(code=1000, reason="Call ended normally")
                except Exception as e:
                    log.error(f"Error closing OpenAI WebSocket: {e}")
        except Exception as e:
            log.error(f"Error during OpenAI WebSocket cleanup: {e}")
            
        try:
            # Close SignalWire WebSocket
            if call.signalwire_ws:
                try:
                    log.debug("Closing SignalWire WebSocket connection")
                    await call.signalwire_ws.close()
                except Exception as e:
                    log.error(f"Error closing SignalWire WebSocket: {e}")
        except Exception as e:
            log.error(f"Error during SignalWire WebSocket cleanup: {e}")

        # Log transcript after connections are closed
        await self._log_transcripts(transcript, call_duration)

        return True
    
    async def _log_transcripts(self, transcript: List[TranscriptItem], duration: float):
        """
        Log the transcript to a file
        
        Args:
            transcript (List[TranscriptItem]): The transcript to log
            duration (float): Duration of the call
        """
        if not transcript:
            log.warning("Transcript is empty")
            return
        
        # Define transcript log directory
        transcript_dir = Path(os.getenv("TRANSCRIPT_DIR", "transcripts"))
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_file = transcript_dir / f"call_{timestamp}.txt"
        
        # Log transcript to file
        with open(transcript_file, "w") as f:
            f.write(f"Call Duration: {duration:.1f} seconds\n")
            f.write(f"Call Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for t in transcript:
                if t.text.strip():
                    if t.timestamp:
                        f.write(f"{t.timestamp.strftime('%H:%M:%S')} - {t.role.upper()}: {t.text}\n")
                    else:
                        f.write(f"{t.role.upper()}: {t.text}\n")
                    log.debug(f"{t}")
        
        log.info(f"Transcript saved to {transcript_file}")

    async def end_call_function(self, call: Call, reason: str = "Call ended by assistant") -> Dict[str, Any]:
        """
        End an active call
        
        Args:
            call (Call): The call to end
            reason (str): Reason for ending the call
            
        Returns:
            Dict[str, Any]: Result of the operation
        """
        log.info(f"Ending call {call.call_sid} with reason: {reason}")
        
        # Call SignalWire API to end the call
        result = await _call_signalwire_api(
            method='POST',
            endpoint=f"Calls/{call.call_sid}",
            data={
                "Status": "completed"
            }
        )
        
        if "error" not in result:
            # Call our local end_call function to clean up resources
            await self.end_call(call.call_sid)
            return {
                "success": True,
                "message": f"Call ended successfully: {reason}"
            }
        else:
            return {
                "success": False,
                "message": f"Failed to end call: {result.get('error', 'Unknown error')}"
            }
    
    async def forward_call_function(self, call: Call) -> Dict[str, Any]:
        """
        Forward an active call to another number
        
        Args:
            call (Call): The call to forward
            
        Returns:
            Dict[str, Any]: Result of the operation
        """
        if not FORWARDING_NUMBER:
            return {
                "success": False,
                "message": "Forwarding number not set"
            }

        log.info(f"Forwarding call {call.call_sid} to {FORWARDING_NUMBER}")
        
        parsed_num = PhoneNumber.parse(FORWARDING_NUMBER)
        if not parsed_num:
            return {
                "success": False,
                "message": "Invalid forwarding number"
            }
        
        # Call SignalWire API to forward the call
        result = await _call_signalwire_api(
            method='POST',
            endpoint=f"Calls/{call.call_sid}",
            data={
                "Url": f"https://{SIGNALWIRE_CONFIG['space_url']}/api/forward?to={FORWARDING_NUMBER}",
                "Method": "POST"
            }
        )
        
        if "error" not in result:
            return {
                "success": True,
                "message": f"Call forwarded successfully to {FORWARDING_NUMBER}"
            }
        else:
            return {
                "success": False,
                "message": f"Failed to forward call: {result.get('error', 'Unknown error')}"
            }

    @property
    def user_contact(self) -> Contact:
        if not self._user_contact:
            log.critical("User contact not initialized")
            raise Exception("User contact not initialized")
        return self._user_contact


async def main():
    """
    Start the server
    """
    config_params = {
        'app': app,
        'host': HOST,
        'port': PORT,
        'log_level': "info"
    }
    
    config_params['ssl_keyfile'] = str(SSL_KEY_PATH)
    config_params['ssl_certfile'] = str(SSL_CERT_PATH)
    
    config = uvicorn.Config(**config_params)
    server = uvicorn.Server(config)

    await server.serve()

if __name__ == '__main__':
    asyncio.run(main())