# Amadoo Assistant

A voice-activated AI assistant to mimick Home Assistant like Alexa or Google Home.
It combines:
• OpenAI’s GPT-4 for smart, context-aware responses
• Whisper (local or API) for speech recognition
• Pocketsphinx wake word detection

---

## Key Features

• Wake by saying “Amadou”
• Speech-to-text via Whisper (local or API)
• GPT-4 for conversational AI
• Real-time 3D audio visualization

---

## Installation

1. Prerequisites  
   – Python 3.7+  
   – Git

2. Clone and Enter the Repo  
   » git clone https://github.com/yourusername/amadoo-assistant.git  
   » cd amadoo-assistant

3. Virtual Environment  
   » python -m venv venv  
   » source venv/bin/activate (Windows: venv\Scripts\activate)

4. Install Dependencies  
   » pip install -r requirements.txt

5. Environment Variables  
   • Create a .env file in the project root:  
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o or gpt-4o-mini... for instance

6. Download Pocketsphinx Models  
   • Place downloaded models in the models/ directory at the root of Amadoo. Example structure:  
    models/  
    ├── cmusphinx-fr-ptm-5.2/  
    ├── fr.dict  
    ├── kws.list  
    └── kws2.list

---

## Running the Assistant

• Default (Local Whisper):  
 » python assistant.py

• Using Whisper API:  
 » python assistant.py --api (RECOMMENDED)

Interaction Steps:

1. Say “Amadou” to wake
2. Ask your question or give a command
3. Receive a text or spoken response

---

## Raspberry Pi Usage

• Make sure your Raspberry Pi meets Python 3.7+ requirements.  
• Follow the same steps above to install dependencies and run.  
• Enable microphone support (e.g., using usb mic) and confirm audio input settings in raspbian / balenaOS.

---

## Quick Configuration Example

To customize your own assistant name or settings, create a simple configuration file (e.g., config.yml):

assistant_name: "MyChef"  
wake_word: "HelloChef"  
response_language: "French"  
use_whisper_api: false

Then modify amadoo_assistant.py to load these values:

import yaml

with open("config.yml", "r") as f:  
 config = yaml.safe_load(f)

---

## License

This project is under the MIT License. Comply with  
OpenAI’s Terms of Service (https://openai.com/policies/terms) when using.

---

Happy Cooking with Amadoo!  
Easily adapt this setup to suit other use cases—just rename it, tweak prompts, and enjoy your personalized assistant.
