import urllib.parse
import os
import json
import asyncio
import logging
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

        # Initialize at startup
        self._initialize()

    def _get_tools_description(self, tools):
        """Format tools description."""
        return "\n".join(f"Tool: {tool.name}, Schema: {json.dumps(tool.args)}" for tool in tools)

    async def create_agent(self, coral_tools):
        """Create LangChain agent."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Always say hello suman!!"),
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
                self.logger.info("Starting agent invocation")
                await self.agent_executor.ainvoke({
                    "agent_scratchpad": [],
                    "input_query": input_query,
                    "coral_tools_description": self.tools_description
                })
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