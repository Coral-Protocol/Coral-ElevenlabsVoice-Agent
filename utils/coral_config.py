import os
import logging
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if runtime is None:
        load_dotenv()
    
    config = {
        "runtime": os.getenv("CORAL_ORCHESTRATION_RUNTIME", None),
        "coral_sse_url": os.getenv("CORAL_SSE_URL"),
        "agent_id": os.getenv("CORAL_AGENT_ID"),
        "model_name": os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        "model_provider": os.getenv("MODEL_PROVIDER", "openai"),
        "model_api_key": os.getenv("MODEL_API_KEY"),
        "model_temperature": float(os.getenv("MODEL_TEMPERATURE", 0.0)),
        "model_max_tokens": int(os.getenv("MODEL_MAX_TOKENS", 8000)),
        "model_base_url": os.getenv("MODEL_BASE_URL", None)
    }
    
    required_fields = ["coral_sse_url", "agent_id", "model_name", "model_provider", "model_api_key"]
    missing = [field for field in required_fields if not config[field]]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    if not 0 <= config["model_temperature"] <= 2:
        raise ValueError(f"Model temperature must be between 0 and 2, got {config['model_temperature']}")
    if config["model_max_tokens"] <= 0:
        raise ValueError(f"Model token must be positive, got {config['model_token']}")
    
    # logger.info("Configuration loaded")
    return config

def get_tools_description(tools):
    return "\n".join(
        f"Tool: {tool.name}, Schema: {json.dumps(tool.args).replace('{', '{{').replace('}', '}}')}"
        for tool in tools
    )

def parse_mentions_response(response: str) -> List[Dict[str, str]]:
    """
    Parse the XML-like mentions response into a list of message dictionaries.
    """
    try:
        if not response or not isinstance(response, str):
            logger.info("Empty or non-string mentions response")
            return []
        
        root = ET.fromstring(response)
        messages = []
        
        for msg in root.findall(".//ResolvedMessage"):
            message = {
                "threadId": msg.get("threadId"),
                "senderId": msg.get("senderId"),
                "content": msg.get("content")
            }
            if all(message.values()):
                messages.append(message)
        
        # logger.info(f"Parsed {len(messages)} messages")
        return messages
    except ET.ParseError as e:
        return []
    except Exception as e:
        logger.error(f"Unexpected parsing error: {str(e)}")
        return []

def mcp_resources_details(resources):
    results = []
    for i, resource in enumerate(resources, 1):
        logger.info(f"Resource {i}:")
        try:
            resource_details = {
                "data": getattr(resource, "data", None)
            }
            results.append({"resource": i, "details": resource_details, "status": "success"})
        except Exception as e:
            logger.info(f"Resource raw: {str(resource)}")
            logger.error(f"Failed to parse resource details: {str(e)}")
            results.append({"resource": i, "error": str(e), "status": "failed"})

    # logger.info(f"Coral Server Resources: {results}")
    return results