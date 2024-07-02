from enum import Enum
from openai import AsyncOpenAI, BaseModel
import os
import logging
from typing import List


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

# Configure OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


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
        logger.error(f"OpenAI API error: {str(e)}")
        return f"I'm sorry, but I encountered an error: {str(e)}"
