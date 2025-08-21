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

    def call_coral_agent(self, parameter: str = None, *args, **kwargs):
        """Async function to call CoralAgent with the latest transcript."""
        if not hasattr(self, 'coral_agent') or not isinstance(self.coral_agent, CoralAgent):
            return "Error: CoralAgent not properly initialized"
        
        if self.latest_transcript and self.latest_transcript.strip():
            coral_agent_input = self.latest_transcript.strip()
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
        else:
            coral_agent_output = "No transcript available"
        
        return coral_agent_output

    def update_transcript(self, transcript):
        """Callback to update the latest transcript."""
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
    signal.signal(signal.SIGINT, lambda sig, frame: conversation.end_session())

def main():
    """Main function to set up and start the conversation."""
    coral_agent_id, elevenlabs_agent_id, elevenlabs_api_key = load_environment()

    if not coral_agent_id or not elevenlabs_agent_id or not elevenlabs_api_key:
        print("Error: CORAL_AGENT_ID, ELEVENLABS_AGENT_ID, and ELEVENLABS_API_KEY must be set.")
        return
    
    # Create conversation manager to handle transcript and callbacks
    conversation_manager = ConversationManager()
    
    # Set up client tools and register the call_coral_agent function
    client_tools = setup_client_tools(conversation_manager)
    
    # Initialize dynamic variables
    dynamic_vars = {
        "agent_name": coral_agent_id,
    }
    
    # Initialize conversation
    conversation = initialize_conversation(elevenlabs_agent_id, elevenlabs_api_key, dynamic_vars, client_tools, conversation_manager)

    # Set up signal handler for termination
    handle_interrupt(conversation)
    
    # Start the conversation session
    conversation.start_session()
    
    # Wait for the session to end
    conversation_id = conversation.wait_for_session_end()
    if conversation_id:
        print(f"Conversation ended. Conversation ID: {conversation_id}")

if __name__ == "__main__":
    main()