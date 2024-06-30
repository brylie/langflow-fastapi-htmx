import pytest
from unittest.mock import AsyncMock, patch
from app.chat_gpt_client import get_chat_response_with_history, Message, MessageRole
import os

# Fixture for chat history
@pytest.fixture
def chat_history():
    return [
        Message(role=MessageRole.user, content="Hello"),
        Message(role=MessageRole.assistant, content="Hi! How can I help you today?")
    ]

# Fixture for environment variables
@pytest.fixture
def load_env_variables(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")

@pytest.mark.asyncio
@patch("app.chat_gpt_client.client")
async def test_get_chat_response_with_history_success(mock_client, chat_history, load_env_variables):
    mock_client.chat.completions.create = AsyncMock(return_value=MockResponse())
    response = await get_chat_response_with_history(chat_history)
    assert response == "Mocked response content"

@pytest.mark.asyncio
@patch("app.chat_gpt_client.client")
async def test_get_chat_response_with_history_api_error(mock_client, chat_history, load_env_variables):
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
    response = await get_chat_response_with_history(chat_history)
    assert "I'm sorry, but I encountered an error: API error" in response

@pytest.mark.asyncio
@patch("app.chat_gpt_client.client")
async def test_get_chat_response_with_history_empty_message(mock_client, load_env_variables):
    response = await get_chat_response_with_history([])
    assert response == "Mocked response content"  # Assuming the API handles empty histories gracefully

@pytest.mark.parametrize("message_content,expected", [
    (["Hello"], "Mocked response content"),
    ([], "Mocked response content"),  # Testing with an empty history
])
@pytest.mark.asyncio
@patch("app.chat_gpt_client.client")
async def test_get_chat_response_with_history_parameterized(mock_client, message_content, expected, load_env_variables):
    mock_client.chat.completions.create = AsyncMock(return_value=MockResponse())
    messages = [Message(role=MessageRole.user, content=content) for content in message_content]
    response = await get_chat_response_with_history(messages)
    assert response == expected

class MockResponse:
    def __init__(self):
        self.choices = [MockChoice()]

class MockChoice:
    def __init__(self):
        self.message = MockMessage()

class MockMessage:
    def __init__(self):
        self.content = "Mocked response content"
