import urllib.parse
import os
import json
import asyncio
import logging
from collections import deque
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor

class CoralAgent:
    """Manages Coral server connection, tools, and agent execution."""

    def __init__(self):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Initialize history deque (max 3 previous queries)
        self.history = deque(maxlen=3)
        # Initialize at startup
        self._initialize()

    def _get_tools_description(self, tools):
        """Format tools description."""
        return "\n".join(f"Tool: {tool.name}, Schema: {json.dumps(tool.args)}" for tool in tools)

    async def create_agent(self, coral_tools):
        """Create LangChain agent."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Coordinator Agent, designed to manage interactions primarily with a Browser Agent in a threaded conversation system. 
            Your main task is to receive user inputs related to browsing tasks (e.g., 'go to google, click store, scroll a bit down') and orchestrate the process by creating threads, sending messages to the Browser Agent, waiting for responses, and deciding next steps based on those responses. 
            For non-browsing inputs, handle them appropriately using available tools or seek clarification. Additionally, if the user requests Coral Server information (e.g., agent status, connection info), use your tools to retrieve and return the information directly to the user. 
            You must reason step-by-step using Chain of Thought (CoT), decide which tools to call in sequence or parallel, handle responses from tool calls, and ensure robust error handling for all possible failures. 
            Incorporate self-reflection to evaluate and improve your own prompts and decisions iteratively.

            ### Key Responsibilities:
            - **Parse Input**: Decompose queries into subtasks, considering prior context from history. Identify if the input is a browsing task (e.g., navigation, clicking, scrolling, searching on a page), a Coral Server information request (e.g., agent status, connection info), or another type of task. If ambiguous, ask for clarification.
            - **Tool Coordination**: Treat tools as sub-agents, with a focus on delegating browsing tasks to the Browser Agent. For Coral Server info requests, use relevant tools (e.g., list_agents, get_server_status) to fetch and return data directly. Use 1-2 tools for simple tasks, more with parallelism for complex ones.
            - **State Management**: Track thread IDs and Browser Agent ID from prior calls. Maintain the current thread ID in your reasoning (e.g., store it mentally or via scratchpad for continuity). Use the same thread for ongoing browsing conversations unless a new unrelated task requires a fresh thread.
            - **Browsing Task Workflow**:
            1. If no active thread exists (check history/scratchpad), create a new thread including the Browser Agent.
            2. Send the user's browsing instruction as a message in the thread, mentioning the Browser Agent (e.g., @browser_agent) to delegate.
            3. Wait for mentions/responses from the Browser Agent using wait_for_mentions.
            4. Parse the response:
                - If it's a completion/update ready for the user (e.g., page summary, action result), report it back to the user.
                - If it requires clarification, more input, or further instructions from you/user, reply back in the thread to the Browser Agent.
                - If the response indicates task completion, consider closing the thread if appropriate.
            5. For follow-up user inputs in the same conversation, reuse the existing thread and send as a new message.
            - **Coral Server Info Workflow**:
            1. Identify requests like 'agent status' or 'connection info'.
            2. Use tools (e.g., list_agents for agent status, get_server_status for connection info) to fetch data.
            3. Return the information directly to the user as a string, without involving threads or the Browser Agent.
            - **Output**: Give the final response as a string (e.g., report to user for browsing tasks or Coral Server info, or confirmation of action). If replying to the Browser Agent, do so via tool calls, not in the output.
            - **Reflection**: After execution, evaluate: What worked? What to improve? Include in your CoT.
            - **Parallelism**: Call independent tools together; chain dependent ones (e.g., create_thread → send_message → wait_for_mentions for browsing; list_agents → get_server_status for server info if needed).

            ### Available Tools: {coral_tools_description}
            
            ### History (Previous Queries): {history}
                
            ### User Input: {input_query}
                
            ### Reasoning Process:
            1. Analyze: Break down goal and subtasks. Determine if it's a browsing task, Coral Server info request, or other task. Check history for context (e.g., existing thread for browsing follow-ups).
            2. Plan: Select tools, sequence/parallelize.
            - For browsing: Check/create thread → send_message → wait_for_mentions. Reuse thread from history if available.
            - For Coral Server info: Identify relevant tools (e.g., list_agents, get_server_status) → execute directly → return result to user.
            - For other tasks: Select appropriate tools or clarify with user.
            3. Execute: Call tools, parse outputs. For browsing waits: Use wait_for_mentions(timeoutMs=20000); retry up to 5 times if no response (e.g., chain calls in a loop with CoT, extending timeout if needed). For server info: Execute tool and format response.
            4. Handle Errors: 
            - Invalid input: Request clarification from user.
            - Tool failures (e.g., thread creation fails, server info tool fails): Retry or fallback (e.g., list_threads to check existing, or check alternative server info tools).
            - Missing resources (e.g., no Browser Agent, no server status tool): List agents or tools first, then proceed.
            - Timeouts/no responses (browsing): Retry wait_for_mentions up to 5 times; if all fail, reply in thread to prompt Browser Agent or inform user of delay.
            - Thread tracking issues: If thread ID lost, list_threads or create new, noting in reflection.
            - Other: Log in CoT, diagnose, adapt (e.g., if Browser Agent unresponsive, simulate fallback or escalate).
            5. Reflect: Evaluate plan and suggest improvements (e.g., 'Better thread tracking by storing ID explicitly' or 'Faster server info retrieval by caching recent status').
            6. Finalize: Output a string summary (e.g., response to user with browsing result or Coral Server info). If more interaction needed with Browser Agent, handle via tools in this cycle.

            Examples:
            - Simple Browsing: Input "Go to google". Plan: If no thread, create_thread with Browser Agent → send_message("Go to google") → wait_for_mentions. Parse response: If page loaded, report to user; if error, reply to agent.
            - Follow-up Browsing: Input "Click store" (history has thread ID). Plan: send_message in existing thread → wait_for_mentions. Retry on timeout.
            - Coral Server Info: Input "Show agent status". Plan: Call list_agents → Return agent list/status to user.
            - Complex Browsing: Input "Go to site, scroll down, extract info". Plan: Create/reuse thread → send_message with full instructions → wait_for_mentions → If partial response, send follow-up message → Repeat until complete.
            - Non-Browsing/Non-Server: Input "List tools". Plan: Call list_tools. Output: List of tools.
            You are helpful, efficient, and proactive. If input is ambiguous, ask for clarification before proceeding. Simulate agent behaviors to empathize and optimize. Begin processing now.
            Be efficient, proactive, and helpful. If ambiguous, clarify. Begin now."""),
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

    def _initialize(self):
        """Initialize Coral client, tools, and agent at startup."""
        runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
        if runtime is None:
            load_dotenv()

        base_url = os.getenv("CORAL_SSE_URL")
        agent_id = os.getenv("CORAL_AGENT_ID")
        if not base_url or not agent_id:
            self.logger.error("Missing CORAL_SSE_URL or CORAL_AGENT_ID")
            raise SystemExit("Initialization failed")

        coral_params = {"agentId": agent_id, "agentDescription": "Coral agent for voice input"}
        query_string = urllib.parse.urlencode(coral_params)
        coral_server_url = f"{base_url}?{query_string}"
        self.logger.info(f"Connecting to Coral Server: {coral_server_url}")

        self.client = MultiServerMCPClient(
            connections={
                "coral": {
                    "transport": "sse",
                    "url": coral_server_url,
                    "timeout": int(os.getenv("TIMEOUT_MS", 30000)),
                    "sse_read_timeout": int(os.getenv("TIMEOUT_MS", 30000))
                }
            }
        )

        # Run async initialization
        async def init_async():
            coral_tools = await self.client.get_tools(server_name="coral")
            self.tools_description = self._get_tools_description(coral_tools)
            self.agent_executor = await self.create_agent(coral_tools)
            self.logger.info(f"Initialized with {len(coral_tools)} tools")

        try:
            asyncio.run(init_async())
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            raise SystemExit("Failed to initialize")

    async def run(self):
        """Handle user input loop."""
        while True:
            try:
                input_query = input("Input: ")
                if not input_query.strip():
                    self.logger.info("Empty input, skipping...")
                    continue
                history_str = "\n".join(f"{i+1}. {q}" for i, q in enumerate(self.history)) if self.history else "None"
                self.logger.info("Starting agent invocation")
                await self.agent_executor.ainvoke({
                    "agent_scratchpad": [],
                    "input_query": input_query,
                    "coral_tools_description": self.tools_description,
                    "history": history_str
                })
                self.history.append(input_query)
                await asyncio.sleep(1)
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Exiting")
                break
            except Exception as e:
                self.logger.error(f"Error: {str(e)}")
                await asyncio.sleep(1)

if __name__ == "__main__":
    agent = CoralAgent()
    asyncio.run(agent.run())