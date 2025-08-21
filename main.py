import os
import signal
import asyncio
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ConversationInitiationData, ClientTools
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from dotenv import load_dotenv
from utils.coral_agent import CoralAgent

def load_environment():
    """Load environment variables, optionally from a .env file."""
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if runtime is None:
        load_dotenv()
    return os.getenv("CORAL_AGENT_ID"), os.getenv("ELEVENLABS_AGENT_ID"), os.getenv("ELEVENLABS_API_KEY")

class ConversationManager:
    """Manages conversation state and callbacks, including the latest user transcript."""
    def __init__(self):
        self.latest_transcript = None
        self.coral_agent = CoralAgent()

    def call_coral_agent(self, input_text: str = None, *args, **kwargs):
        """Process the provided input (from text or voice) or fall back to latest transcript with CoralAgent."""
        if not hasattr(self, 'coral_agent') or not isinstance(self.coral_agent, CoralAgent):
            return "Error: CoralAgent not properly initialized"

        # Handle case where input_text is a dictionary (from voice input)
        if isinstance(input_text, dict):
            input_text = input_text.get("transcript", "")

        # Use input_text if provided and valid, otherwise fall back to latest_transcript
        coral_agent_input = None
        if input_text and input_text.strip():
            coral_agent_input = input_text.strip()
        elif self.latest_transcript and self.latest_transcript.strip():
            coral_agent_input = self.latest_transcript.strip()
        else:
            return "Error: No valid input or transcript available"

        # Process the input with CoralAgent
        history_str = "\n".join(f"{i+1}. {q}" for i, q in enumerate(self.coral_agent.history)) if self.coral_agent.history else "None"
        
        try:
            result = asyncio.run(self.coral_agent.agent_executor.ainvoke({
                "agent_scratchpad": [],
                "input_query": coral_agent_input,
                "coral_tools_description": self.coral_agent.tools_description,
                "history": history_str
            }))
            coral_agent_output = result.get("output", "No response from CoralAgent")
            self.coral_agent.history.append(coral_agent_input)
        except Exception as e:
            coral_agent_output = f"Error in CoralAgent: {str(e)}"
        
        return coral_agent_output

    def update_transcript(self, transcript):
        """Callback to update the latest transcript from voice input."""
        self.latest_transcript = transcript

def setup_client_tools(conversation_manager):
    """Set up client tools, registering the call_coral_agent function."""
    client_tools = ClientTools()
    client_tools.start()
    client_tools.register("call_coral_agent", conversation_manager.call_coral_agent, is_async=False)
    return client_tools

def initialize_conversation(elevenlabs_agent_id, elevenlabs_api_key, dynamic_vars, client_tools, conversation_manager):
    """Initialize the ElevenLabs conversation with dynamic variables and callbacks."""
    elevenlabs = ElevenLabs(api_key=elevenlabs_api_key)

    config = ConversationInitiationData(
        dynamic_variables=dynamic_vars
    )
    
    conversation = Conversation(
        elevenlabs,
        elevenlabs_agent_id,
        config=config,
        requires_auth=bool(elevenlabs_api_key),
        audio_interface=DefaultAudioInterface(),
        client_tools=client_tools,
        callback_agent_response=lambda response: print(f"Agent: {response}"),
        callback_agent_response_correction=lambda original, corrected: print(f"Agent: {original} -> {corrected}"),
        callback_user_transcript=conversation_manager.update_transcript,
        callback_latency_measurement=lambda latency: print(f"Latency: {latency}ms"),
    )
    
    return conversation

def handle_interrupt(conversation):
    """Set up signal handler for graceful termination."""
    def signal_handler(sig, frame):
        conversation.end_session()
        print("\nVoice session terminated.")
        raise KeyboardInterrupt  # Allow main loop to catch and return to text mode
    signal.signal(signal.SIGINT, signal_handler)

async def run_voice_session(elevenlabs_agent_id, elevenlabs_api_key, dynamic_vars, client_tools, conversation_manager):
    """Run the voice conversation with a 2-minute timeout."""
    conversation = initialize_conversation(elevenlabs_agent_id, elevenlabs_api_key, dynamic_vars, client_tools, conversation_manager)
    handle_interrupt(conversation)
    
    try:
        conversation.start_session()
        # Wait for session to end or timeout after 2 minutes (120 seconds)
        conversation_id = await asyncio.wait_for(conversation.wait_for_session_end(), timeout=120)
        if conversation_id:
            print(f"Voice session ended. Conversation ID: {conversation_id}")
    except asyncio.TimeoutError:
        print("Voice session timed out after 2 minutes.")
        conversation.end_session()
    except KeyboardInterrupt:
        pass  # Handled by signal_handler
    except Exception as e:
        print(f"Error in voice session: {str(e)}")
        conversation.end_session()

def main():
    """Main function to handle text and voice input based on user choice."""
    coral_agent_id, elevenlabs_agent_id, elevenlabs_api_key = load_environment()

    if not coral_agent_id or not elevenlabs_agent_id or not elevenlabs_api_key:
        print("Error: CORAL_AGENT_ID, ELEVENLABS_AGENT_ID, and ELEVENLABS_API_KEY must be set.")
        return
    
    # Create conversation manager to handle transcript and callbacks
    conversation_manager = ConversationManager()
    
    # Set up client tools for voice mode
    client_tools = setup_client_tools(conversation_manager)
    
    # Initialize dynamic variables for voice mode
    dynamic_vars = {
        "agent_name": coral_agent_id,
    }
    
    # Main loop for text input, with option to switch to voice
    while True:
        try:
            user_input = input("Enter your query (or 'v' for voice, 'quit' to exit): ").strip()
            if user_input.lower() == 'quit':
                print("Exiting program.")
                break
            elif user_input.lower() == 'v':
                print("Switching to voice mode (2-minute timeout). Speak now or press Ctrl+C to return to text mode.")
                # Run voice session in async context
                asyncio.run(run_voice_session(elevenlabs_agent_id, elevenlabs_api_key, dynamic_vars, client_tools, conversation_manager))
                print("Returning to text mode.")
            elif user_input:
                # Process text input
                response = conversation_manager.call_coral_agent(user_input)
                print(f"Agent: {response}")
            else:
                print("Error: No input provided")
        except KeyboardInterrupt:
            print("\nReturning to text mode.")
            continue
        except Exception as e:
            print(f"Error: {str(e)}")
            continue

if __name__ == "__main__":
    main()
