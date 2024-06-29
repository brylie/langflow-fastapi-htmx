from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict

from llm_client import get_chat_response  # Import the function from llm_client.py

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Simulating a database with an in-memory list
chat_history: List[Dict[str, str]] = []


class Message(BaseModel):
    role: str
    content: str


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "chat.html", {"request": request, "chat_history": chat_history}
    )


@app.post("/chat")
async def chat(message: str = Form(...)) -> HTMLResponse:
    # Add user message to chat history
    chat_history.append({"role": "user", "content": message})

    # Get response from LLM
    bot_response = await get_chat_response(message)

    # Add bot response to chat history
    chat_history.append({"role": "bot", "content": bot_response})

    # Return HTML for bot response
    return HTMLResponse(f"""
        <div class="message bot-message">
            <p>{bot_response}</p>
        </div>
    """)


@app.get("/api/chat_history")
async def get_chat_history() -> List[Dict[str, str]]:
    return chat_history


# Optional: Add a route to clear chat history (for testing/demo purposes)
@app.post("/api/clear_history")
async def clear_history() -> Dict[str, str]:
    chat_history.clear()
    return {"message": "Chat history cleared"}
