from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
import markdown2

from llm_client import get_chat_response

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Simulating a database with an in-memory list
chat_history: List[Dict[str, str]] = []


class Message(BaseModel):
    role: str
    content: str


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
    chat_history.append({"role": "user", "content": message})

    # Get response from LLM
    bot_response = await get_chat_response(message)

    # Render Markdown to HTML (with safety features)
    bot_response_html = markdown2.markdown(bot_response, safe_mode="escape")

    # Add bot response to chat history
    chat_history.append({"role": "bot", "content": bot_response_html})

    # Return HTML for bot response
    return HTMLResponse(f"""
        <div class="message bot-message">
            {bot_response_html}
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
