from enum import Enum
from openai import AsyncOpenAI, BaseModel
import os
import logging
from typing import List
import agentops
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize AgentOps
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
if not AGENTOPS_API_KEY:
    raise ValueError("AGENTOPS_API_KEY not found in environment variables")

agentops.init(AGENTOPS_API_KEY)

CHAT_GPT_DEFAULT_MODEL = os.getenv("CHAT_GPT_MODEL", "gpt-4o")
CHAT_GPT_DEFAULT_TEMPERATURE = float(os.getenv("CHAT_GPT_TEMPERATURE", "0.7"))
CHAT_GPT_DEFAULT_MAX_TOKENS = int(os.getenv("CHAT_GPT_MAX_TOKENS", "1500"))

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"

class Message(BaseModel):
    role: MessageRole
    content: str

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@agentops.record_function('configure_openai_client')
def configure_openai_client():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return client

client = configure_openai_client()

@agentops.record_function('get_chat_response_with_history')
async def get_chat_response_with_history(
    messages: List[Message],
    system_prompt: str = "You are a helpful assistant that always answers questions.",
    model: str = CHAT_GPT_DEFAULT_MODEL,
    temperature: float = CHAT_GPT_DEFAULT_TEMPERATURE,
    max_tokens: int = CHAT_GPT_DEFAULT_MAX_TOKENS,
) -> str:
    """
    Asynchronous function to get a chat response from OpenAI's ChatGPT, considering chat history.

    :param messages: List of previous messages, each a Message object with 'role' and 'content'
    :param system_prompt: The system message to set the behavior of the assistant
    :param model: The GPT model to use
    :param temperature: Controls randomness (0 to 1)
    :param max_tokens: Maximum number of tokens in the response
    :return: The assistant's response as a string
    """
    try:
        full_messages = [Message(role=MessageRole.system, content=system_prompt)] + [
            Message(role=msg.role.value, content=msg.content) for msg in messages
        ]
        response = await client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_message = f"OpenAI API error: {str(e)}"
        logger.error(error_message)
        agentops.log_error(error_message)
        return f"I'm sorry, but I encountered an error: {str(e)}"

@agentops.record_function('main')
async def main():
    # Example usage
    messages = [
        Message(role=MessageRole.user, content="Hello, how are you?"),
        Message(role=MessageRole.assistant, content="I'm doing well, thank you for asking. How can I assist you today?"),
        Message(role=MessageRole.user, content="Can you tell me about the weather?"),
    ]
    response = await get_chat_response_with_history(messages)
    print(f"Assistant's response: {response}")

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except Exception as e:
        agentops.log_error(str(e))
    finally:
        agentops.end_session('Success')