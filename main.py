import urllib.parse
from dotenv import load_dotenv
import os, json, asyncio, traceback
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
import signal
import sys
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from websockets.exceptions import ConnectionClosedOK
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
import logging
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_tools_description(tools):
    return "\n".join(
        f"Tool: {tool.name}, Schema: {json.dumps(tool.args).replace('{', '{{').replace('}', '}}')}"
        for tool in tools
    )

def run_agent():
    elevenlabs_agent_id = os.getenv("ELEVENLABS_AGENT_ID")
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
    
    if not elevenlabs_agent_id or not elevenlabs_api_key:
        raise ValueError("ELEVENLABS_AGENT_ID or ELEVENLABS_API_KEY is not set in the .env file")

    elevenlabs = ElevenLabs(api_key=elevenlabs_api_key)
    
    latest_action = [None]
    conversation_active = [True]
    user_input_received = threading.Event()
    
    def signal_handler(sig, frame):
        print("Received SIGINT, ending session...")
        conversation_active[0] = False
        if 'conversation' in locals():
            conversation.end_session()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    def start_conversation():
        latest_user_transcript = [None]
        response_completed = [False]
    
        def process_agent_response(response):
            print(f"Agent: {response}")
            print(f"Agent response completed, ready for next input...")
            response_completed[0] = True
        
        def update_user_transcript(transcript):
            print(f"User: {transcript}")
            if transcript and transcript.strip():
                latest_user_transcript[0] = transcript
                latest_action[0] = transcript
                user_input_received.set()
                print("Valid transcript received, processing...")
                
        conversation = Conversation(
            client=elevenlabs,
            agent_id=elevenlabs_agent_id,
            requires_auth=bool(elevenlabs_api_key),
            audio_interface=DefaultAudioInterface(),
            callback_agent_response=process_agent_response,
            callback_agent_response_correction=lambda original, corrected: print(f"Agent: {original} -> {corrected}"),
            callback_user_transcript=update_user_transcript,
            callback_latency_measurement=lambda latency: print(f"Latency: {latency}ms"),
        )
        return conversation
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            conversation = start_conversation()
            conversation.start_session()
            
            print("Listening for user input...")
            user_input_received.wait(timeout=60)
            
            if latest_action[0]:
                print(f"Received user input: {latest_action[0]}")
                time.sleep(1)
            
            conversation.end_session()
            conversation_id = conversation.wait_for_session_end()
            print(f"Conversation ID: {conversation_id}")
            break
            
        except ConnectionClosedOK as e:
            print(f"WebSocket connection closed: {e}. Attempt {attempt + 1}/{max_attempts}")
            if attempt == max_attempts - 1:
                print("Max reconnection attempts reached. Exiting.")
                sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            break
    
    return latest_action[0] if latest_action[0] else "No user input received"

async def ask_human_tool(question: str) -> str:
    print(f"Agent asks: {question}")
    response = run_agent()
    return response

async def tell_human_tool(statement: str) -> str:
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    if not voice_id:
        raise ValueError("ELEVENLABS_VOICE_ID is not set in the .env file")

    audio = client.text_to_speech.convert(
        text=statement,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    play(audio)

    print(f"Agent said: {statement}")
    return "Message delivered to user."

async def create_agent(coral_tools, agent_tools, runtime):
    coral_tools_description = get_tools_description(coral_tools)
    
    if runtime is not None:
        agent_tools_for_description = [
            tool for tool in coral_tools if tool.name in agent_tools
        ]
        agent_tools_description = get_tools_description(agent_tools_for_description)
        combined_tools = coral_tools + agent_tools_for_description
        user_request_tool = "request_question"
        user_answer_tool = "answer_question"
        print(agent_tools_description)
    else:
        agent_tools_description = get_tools_description(agent_tools)
        combined_tools = coral_tools + agent_tools
        user_request_tool = "ask_human"
        user_answer_tool = "tell_human"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""Your name is Coral, an agent which can interact with the tools and agents from Coral Server to fulfill requests of the user.

            You are also using your own `{user_request_tool}` and `{user_answer_tool}` tool to communicate with the user.

            **You MUST NEVER finish the chain**

            Follow these steps in order:
            1. Use `list_agents` to list all connected agents and get their descriptions.
            2. Use tool `{user_answer_tool}` to tell user "Hello, I am Coral. How can I assist you?"
            3. Use tool `{user_request_tool}` to listen to the user's request.
            4. Understand the user's intent and decide which agent(s) are needed based on their descriptions.
            5. If the user requests Coral Server information (e.g., agent status, connection info), use your tools to retrieve and return the information directly to the user, then go back to Step 1.
            6. If fulfilling the request requires multiple agents, then call
                `create_thread ('threadName': , 'participantIds': [ID of all required agents, including yourself])` to create conversation thread.
            7. For each selected agent:
            * **If the required agent is not in the thread, add it by calling `add_participant(threadId=..., 'participantIds': ID of the agent to add)`.**
            * Construct a clear instruction message for the agent.
            * Use **`send_message(threadId=..., content="instruction", mentions=[Receive Agent Id])`.** (NEVER leave `mentions` as empty)
            * Use `wait_for_mentions(timeoutMs=60000)` to receive the agent's response up to 5 times if no message received.
            * Record and store the response for final presentation.
            8. After all required agents have responded, think about the content to ensure you have executed the instruction to the best of your ability and the tools. Make this your response as "answer".
            9. Always respond back to the user by calling `{user_answer_tool}` with the "answer" or error occurred even if you have no answer or error.
            10. Repeat the process from Step 1.

            **You MUST NEVER finish the chain**
            
            These are the list of coral tools: {coral_tools_description}
            These are the list of agent tools: {agent_tools_description}

            **You MUST NEVER finish the chain**"""
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
    agent = create_tool_calling_agent(model, combined_tools, prompt)
    return AgentExecutor(agent=agent, tools=combined_tools, verbose=True)

async def main():
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if runtime is None:
        load_dotenv()

    base_url = os.getenv("CORAL_SSE_URL")
    agentID = os.getenv("CORAL_AGENT_ID")

    coral_params = {
        "agentId": agentID,
        "agentDescription": "Coral agent that takes user input via voice and interacts with other agents to fulfill requests of the user."
    }

    query_string = urllib.parse.urlencode(coral_params)

    CORAL_SERVER_URL = f"{base_url}?{query_string}"
    logger.info(f"Connecting to Coral Server: {CORAL_SERVER_URL}")

    timeout = os.getenv("TIMEOUT_MS", 30000)
    client = MultiServerMCPClient(
        connections={
            "coral": {
                "transport": "sse",
                "url": CORAL_SERVER_URL,
                "timeout": timeout,
                "sse_read_timeout": timeout,
            }
        }
    )
    logger.info("Coral Server Connection Established")

    coral_tools = await client.get_tools(server_name="coral")
    logger.info(f"Coral tools count: {len(coral_tools)}")
    
    if runtime is not None:
        required_tools = ["request_question", "answer_question"]
        available_tools = [tool.name for tool in coral_tools]

        for tool_name in required_tools:
            if tool_name not in available_tools:
                error_message = f"Required tool '{tool_name}' not found in coral_tools. Please ensure that while adding the agent on Coral Studio, you include the tool from Custom Tools."
                logger.error(error_message)
                raise ValueError(error_message)        
        agent_tools = required_tools

    else:
        agent_tools = [
            Tool(
                name="ask_human",
                func=None,
                coroutine=ask_human_tool,
                description="Ask the user a question by speaking it and wait for their spoken response."
            ),
            Tool(
                name="tell_human",
                func=None,
                coroutine=tell_human_tool,
                description="Tell the user something by speaking it, such as the final answer or information."
            )
        ]
    
    agent_executor = await create_agent(coral_tools, agent_tools, runtime)

    while True:
        try:
            logger.info("Starting new agent invocation")
            await agent_executor.ainvoke({"agent_scratchpad": []})
            logger.info("Completed agent invocation, restarting loop")
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error in agent loop: {str(e)}")
            logger.error(traceback.format_exc())
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())