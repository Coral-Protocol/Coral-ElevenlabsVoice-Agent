# Coral Elevenlabs Voice Agent

User Interaction Agent for voice-based user instructions, coordinating multi-agent tasks, and logging conversations via the terminal using ElevenLabs and Coral Protocol.

## Responsibility
Coral Elevenlabs Voice Agent acts as the main interface for receiving user instructions via voice, orchestrating requests among various agents, and ensuring seamless workflow and conversation logging. It leverages ElevenLabs for voice input/output and Coral Protocol for multi-agent coordination.

## Details
- **Frameworks**: LangChain, ElevenLabs
- **Tools used**: Coral MCP Tools, LangChain, ElevenLabs
- **AI model**: Configurable (e.g., GPT-4.1, GROQ-llama-3.3-70b, OpenAI, etc.)
- **Date added**: June 4, 2025
- **License**: MIT

## Setup the Agent

### 1. Clone & Install Dependencies

<details>

```bash
# In a new terminal clone the repository:
git clone <your-repo-url>

# Navigate to the project directory:
cd Coral-ElevenlabsVoice-Agent

# Download and run the UV installer, setting the installation directory to the current one
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=$(pwd) sh

# Create a virtual environment named `.venv` using UV
uv venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# install uv
pip install uv

# Install dependencies from `pyproject.toml` using `uv`:
uv sync
```

</details>

### 2. Configure Environment Variables

Get the API Key:
- [OpenAI](https://platform.openai.com/api-keys)
- [GROQ](https://console.groq.com/keys)
- [ElevenLabs](https://elevenlabs.io/)

<details>

```bash
# Copy the sample environment file and edit as needed
cp .env_sample .env
# Then edit .env to add your API keys and configuration
```
</details>

### 3. ElevenLabs System Prompt

<details>

```
Your name is <code>{{agent_name}}</code>, and you are a friendly, voice-activated conversational AI assistant helping the user. Your role is to listen carefully to the user's spoken input and decide how to respond based on its nature:

- If the user's query is general information, casual chit-chat, simple questions you can answer directly from basic knowledge (e.g., greetings, time, basic facts, math, or everyday conversation), or non-specialized tasks, respond naturally and conversationally without using tools. Be helpful, concise, engaging, and keep responses under 100 words unless more detail is needed.

- If the user's query involves browsing tasks (e.g., navigating websites, clicking links, scrolling, searching on pages like "go to Google and click store"), Coral Server information (e.g., agent status, connection info, list tools/agents), complex computations, data processing, or any task requiring orchestration, tools, or external agents, invoke the 'call_coral_agent' tool with the exact user input as the parameter. Then, integrate the tool's output into a natural response, e.g., "Here's what Coral found: [output]."

Always analyze the input first to classify it. If unsure or ambiguous, handle it directly if simple; otherwise, call the tool. For phrases like "can you" + request, classify based on the rules—don't default to the tool.

**Examples:**

- User: "Hi, how's it going?" → Respond directly: "I'm great, thanks! What's on your mind?"
- User: "What's 2+2?" → Respond directly: "That's 4!"
- User: "Go to example.com and scroll down." → Call 'call_coral_agent' with input, then: "Coral handled that: [output from tool]."
- User: "Show me the agent status." → Call 'call_coral_agent' with input, then: "From Coral: [output]."
- User: "Calculate pi to 10 decimals." → Call 'call_coral_agent' with input, then: "Let me get that from Coral: [output]."
```
</details>

## Run the Agent

You can run in either of the below modes to get your system running.

- The Executable Mode is part of the Coral Protocol Orchestrator which works with [Coral Studio UI](https://github.com/Coral-Protocol/coral-studio).
- The Dev Mode allows the Coral Server and all agents to be separately running on each terminal without UI support.

### 1. Executable Mode

Checkout: [How to Build a Multi-Agent System with Awesome Open Source Agents using Coral Protocol](https://github.com/Coral-Protocol/existing-agent-sessions-tutorial-private-temp) and update the file: `coral-server/src/main/resources/application.yaml` with the details below, then run the [Coral Server](https://github.com/Coral-Protocol/coral-server) and [Coral Studio UI](https://github.com/Coral-Protocol/coral-studio). You do not need to set up the `.env` in the project directory for running in this mode; it will be captured through the variables below.

<details>

For Linux or MAC:

```yaml
registry:
  # ... your other agents
  elevenlabs-voice-agent:
    options:
      - name: "MODEL_API_KEY"
        type: "string"
        description: "API key for the model provider"
      - name: "MODEL_NAME"
        type: "string"
        description: "What model to use (e.g 'gpt-4.1')"
        default: "gpt-4.1"
      - name: "MODEL_PROVIDER"
        type: "string"
        description: "What model provider to use (e.g 'openai', etc)"
        default: "openai"
      - name: "MODEL_MAX_TOKENS"
        type: "string"
        description: "Max tokens to use"
        default: 16000
      - name: "MODEL_TEMPERATURE"
        type: "string"
        description: "What model temperature to use"
        default: "0.3"

    runtime:
      type: "executable"
      command: ["bash", "-c", "<replace with path to this agent>/run_agent.sh main.py"]
      environment:
        - option: "MODEL_API_KEY"
        - option: "MODEL_NAME"
        - option: "MODEL_PROVIDER"
        - option: "MODEL_MAX_TOKENS"
        - option: "MODEL_TEMPERATURE"
```

For Windows, create a powershell command (run_agent.ps1) and run:

```yaml
command: ["powershell","-ExecutionPolicy", "Bypass", "-File", "${PROJECT_DIR}/run_agent.ps1","main.py"]
```

</details>

### 2. Dev Mode

Ensure that the [Coral Server](https://github.com/Coral-Protocol/coral-server) is running on your system and run below command in a separate terminal.

<details>

```bash
# Run the agent using `uv`:
uv run python main.py
```

You can view the agents running in Dev Mode using the [Coral Studio UI](https://github.com/Coral-Protocol/coral-studio) by running it separately in a new terminal.

</details>

## Example

<details>

```bash
# Input (spoken or typed):
Agent: How can I assist you today?

# Output:
The agent will interact with you directly in the console and coordinate with other agents as needed, using voice and text.
```
</details>

## Creator Details
- **Name**: Suman Deb
- **Affiliation**: Coral Protocol
- **Contact**: [Discord](https://discord.com/invite/Xjm892dtt3)
