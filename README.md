# AI Phone Assistant

A cutting-edge demonstration of what's possible when blending OpenAI's latest voice technology with telephony services, creating a virtual phone receptionist that sounds remarkably human.

## Next-Gen Automated Phone Conversations

AI Phone Assistant showcases the potential of AI-powered communication by:

- Answering calls naturally with personalized greetings based on caller identity and time of day
- Understanding context and maintaining meaningful conversations
- Efficiently managing calls by taking messages or forwarding important callers
- Protecting your time by identifying and gracefully ending spam calls
- Documenting everything with complete conversation transcripts

## The Technology Stack

This technology demonstration leverages:

- FastAPI: Providing a robust, high-performance backend
- SignalWire: Handling real-world telephony integration
- OpenAI's GPT-4o: Powering intelligent conversations through the Realtime API
- WebSockets: Enabling truly responsive bidirectional audio streaming

## Key Capabilities

#### Remarkably Natural Conversations

Experience the latest in AI voice technology that sounds increasingly human, understanding nuance and responding appropriately.

#### Smart Call Handling

The assistant intelligently manages your calls based on your preferences and caller identity.

#### Caller Recognition

Identifies known contacts and personalizes interactions accordingly.

#### Spam Protection

Automatically detects and ends unwanted calls so you don't have to.

#### Complete Documentation

Every call is transcribed and archived for your reference.

## Implementation Notes

This project is a technology demonstration that shows what's possible with today's AI and communication technology. While functional, it's designed as an exploratory implementation rather than a commercial product.

The response times and voice quality will vary based on network conditions, API performance, and server capabilities. As a tech demo, it may require optimization for high-volume usage.

## Features

- **Natural Conversation**: Uses OpenAI's GPT-4o with voice capabilities to create natural-sounding conversations.
- **Call Management**: Forward calls or take messages as needed.
- **Personalized Responses**: Greets callers appropriately based on time of day and known contacts.
- **Contact Management**: Simple JSON-based contacts system to recognize callers.
- **Transcript Recording**: Automatically saves transcripts of all calls.
- **Spam Protection**: Identifies and ends spam or telemarketing calls.

## Prerequisites

- Python 3.8+
- [SignalWire](https://signalwire.com/) account with a phone number
- OpenAI API key with access to GPT-4o Realtime
- Certificates from a *trusted* CA

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AidanAllchin/ai-phone-assistant.git
   cd ai-phone-assistant
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:

   ```
   # SignalWire configuration (required)
   SIGNALWIRE_PROJECT_ID=your-project-id
   SIGNALWIRE_TOKEN=your-token
   SIGNALWIRE_SPACE_URL=your-space.signalwire.com

   # OpenAI configuration (required)
   OPENAI_API_KEY=your-openai-key
   OPENAI_MODEL=gpt-4o-realtime-preview-2024-12-17

   # User configuration (optional)
   USER_NAME=Your Name
   USER_TIMEZONE=America/Los_Angeles
   PHONE_NUMBER=+12025551234
   FORWARDING_NUMBER=+12025555678
   VOICE_NAME=shimmer
   ASSISTANT_NAME=Shimmer  # How should the assistant be addressed/introduce themself?

   # Server configuration (optional)
   HOST=0.0.0.0
   PORT=5680
   CERT_DIR=./certs
   SSL_CERT_FILENAME=phone_assistant.crt
   SSL_KEY_FILENAME=phone_assistant.key
   LOG_FILE=phone_assistant.log
   ```

5. Generate SSL certificates (required for the SignalWire connection):

   These must be signed by a trusted certificate authority, and **cannot be self-signed**.

   This can be very easy to do - see the instructions here:

   https://eff-certbot.readthedocs.io/en/stable/using.html#standalone

## Running the Application

Start the application:

```bash
python src/main.py
```

## SignalWire Configuration

1. Log in to your SignalWire dashboard
2. Set up a phone number if you don't have one
3. Add a new Bin under cXML/LaML -> Bins in the following structure:
   ```
   <?xml version="1.0" encoding="UTF-8"?>
   <Response>
   <Connect>
       <Stream url="wss://<server-address>:5680/media-stream">
       <Parameter name="event" value="start" />
       </Stream>
   </Connect>
   </Response>
   ```
4. Configure the number's webhook settings to point to the bin you created:

   - Accept Incoming Calls As: Voice Calls
   - Handle Calls Using: LaML Webhooks
   - When a Call Comes In: <bin name from step 3>
   - Method: POST

## Contact Management

The system uses a simple JSON file (`contacts.json`) to store contacts. Example format:

```json
[
  {
    "first_name": "Jane",
    "last_name": "Doe",
    "phone_numbers": ["+12025551234"],
    "emails": ["jane@example.com"],
    "tags": ["close_friend"]
  },
  {
    "first_name": "John",
    "last_name": "Smith",
    "phone_numbers": ["+13105557890"],
    "emails": ["john@example.com"],
    "tags": ["business", "VIP"]
  }
]
```

## License

MIT

## Credits

This project is a standalone adaptation of a plugin used in a larger personal assistant system by Aidan Allchin.
