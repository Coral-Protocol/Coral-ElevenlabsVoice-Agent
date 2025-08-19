import os
import asyncio
from dotenv import load_dotenv
from io import BytesIO
import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from langchain.chat_models import init_chat_model
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.tools import Tool
import urllib.parse
import json
import logging
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_tools_description(tools):
    """Generate a description of available tools."""
    return "\n".join(
        f"Tool: {tool.name}, Schema: {json.dumps(tool.args).replace('{', '{{').replace('}', '}}')}"
        for tool in tools
    )

async def create_agent(coral_tools):
    """Create and configure the agent with the given tools."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Your name is Coral, an agent that interacts with tools and agents from Coral Server to fulfill user requests and no other thing.
               Do not do web searches or access external content or try to browse the internet.
               When the user asks for browser-related functions, use the 'send_message' tool to send the request directly to the browser agent.
               When asked for list agents just quick name the agents unless asked for details.
               Keep your responses very concise and speak quickly to provide fast replies.
               You have access to these tools: {coral_tools_description}
               User message: {transcription_text}
            """
        ),
        ("placeholder", "{agent_scratchpad}")
    ])

    model = init_chat_model(
        model=os.getenv("MODEL_NAME"),
        model_provider=os.getenv("MODEL_PROVIDER"),
        api_key=os.getenv("MODEL_API_KEY"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0.0)),
        max_tokens=int(os.getenv("MODEL_MAX_TOKENS", 8000)),
        base_url=os.getenv("MODEL_BASE_URL", None)
    )
    agent = create_tool_calling_agent(model, coral_tools, prompt)
    return AgentExecutor(agent=agent, tools=coral_tools, verbose=True)

def record_audio(duration: int = 5, sample_rate: int = 44100) -> BytesIO:
    """Record audio from the microphone and return it as a BytesIO buffer."""
    try:
        logger.info("Recording audio for %d seconds...", duration)
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        logger.info("Recording finished.")
        
        buffer = BytesIO()
        wavfile.write(buffer, sample_rate, audio)
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error("Error recording audio: %s", e)
        return BytesIO()

async def main():
    """Main function to run the voice-based agent."""
    # Validate environment variables
    required_vars = ["ELEVENLABS_API_KEY", "MODEL_NAME", "MODEL_PROVIDER", "MODEL_API_KEY", "CORAL_SSE_URL", "CORAL_AGENT_ID"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error("Missing environment variables: %s", ", ".join(missing_vars))
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

    # Initialize ElevenLabs client
    try:
        elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        logger.info("ElevenLabs client initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize ElevenLabs client: %s", e)
        raise ValueError(f"Failed to initialize ElevenLabs client: {e}")

    # Connect to Coral Server
    base_url = os.getenv("CORAL_SSE_URL")
    agent_id = os.getenv("CORAL_AGENT_ID")
    coral_params = {
        "agentId": agent_id,
        "agentDescription": "Coral agent that takes user input via voice and interacts with other agents to fulfill requests."
    }
    query_string = urllib.parse.urlencode(coral_params)
    coral_server_url = f"{base_url}?{query_string}"
    logger.info("Connecting to Coral Server: %s", coral_server_url)

    timeout = int(os.getenv("TIMEOUT_MS", 30000))
    client = MultiServerMCPClient(
        connections={
            "coral": {
                "transport": "sse",
                "url": coral_server_url,
                "timeout": timeout,
                "sse_read_timeout": timeout,
            }
        }
    )
    logger.info("Coral Server connection established.")

    # Retrieve Coral tools
    coral_tools = await client.get_tools(server_name="coral")
    logger.info("Retrieved %d Coral tools.", len(coral_tools))
    coral_tools_description = get_tools_description(coral_tools)

    # Create agent
    agent_executor = await create_agent(coral_tools)
    logger.info("Agent executor created.")

    try:
        while True:
            # Record audio
            audio_data = record_audio(duration=5)
            if not audio_data.getbuffer().nbytes:
                logger.warning("No audio recorded. Skipping...")
                continue

            # Transcribe audio
            try:
                transcription = elevenlabs.speech_to_text.convert(
                    file=audio_data,
                    model_id="scribe_v1",
                    tag_audio_events=True,
                    language_code="en",
                    diarize=True,
                )
                transcription_text = transcription.text if hasattr(transcription, 'text') else ''
                if not transcription_text.strip():
                    logger.warning("No valid transcription received.")
                    continue
                logger.info("Transcription: %s", transcription_text)
            except Exception as e:
                logger.error("Transcription failed: %s", e)
                continue

            # Process transcription with agent
            try:
                response = await agent_executor.ainvoke({
                    "agent_scratchpad": [],
                    "transcription_text": transcription_text,
                    "coral_tools_description": coral_tools_description
                })
                response_text = response.get('output', 'No response generated.')
                logger.info("Agent response: %s", response_text)
            except Exception as e:
                logger.error("Agent processing failed: %s", e)
                continue

            # Convert response to speech
            try:
                audio = elevenlabs.text_to_speech.convert(
                    text=response_text,
                    voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                )
                logger.info("Playing response audio...")
                play(audio)
            except Exception as e:
                logger.error("Text-to-speech conversion or playback failed: %s", e)
                continue

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.error("Unexpected error: %s", e)
    finally:
        logger.info("Cleaning up resources...")
        # Ensure all async tasks are completed
        await client.aclose()
        logger.info("Resources cleaned up.")

if __name__ == "__main__":
    asyncio.run(main())