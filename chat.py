from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import List, Dict
import markdown2

from chat_gpt_client import get_chat_response_with_history, Message, MessageRole

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# Simulating a database with an in-memory list
chat_history: List[Message] = []


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "chat_history": chat_history,
        },
    )


@app.post("/chat")
async def chat(message: str = Form(...)) -> HTMLResponse:
    # Add user message to chat history
    chat_history.append(Message(role=MessageRole.user, content=message))

    # Prepare chat history for GPT
    gpt_messages = [
        Message(role=message.role, content=message.content)
        for message in chat_history[-5:]
    ]  # Last 5 messages

    # Get response from ChatGPT
    bot_response = await get_chat_response_with_history(gpt_messages)

    # Render Markdown to HTML (with safety features)
    bot_response_html = markdown2.markdown(bot_response, safe_mode="escape")

    # Add bot response to chat history
    chat_history.append(Message(role=MessageRole.assistant, content=bot_response))

    # Return HTML for bot response
    return HTMLResponse(f'<div class="message bot-message">{bot_response_html}</div>')


@app.get("/api/chat_history")
async def get_chat_history() -> List[Dict[str, str]]:
    return [message.model_dump() for message in chat_history]


# Optional: Add a route to clear chat history (for testing/demo purposes)
@app.post("/api/clear_history")
async def clear_history() -> Dict[str, str]:
    chat_history.clear()
    return {"message": "Chat history cleared"}
