import os
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import List, Dict
import markdown2
import uuid
import agentops

from app.chat_gpt_client import get_chat_response_with_history, Message, MessageRole
from app.rag_service import RAGService
from app.vector_store import AstraDBStore

# Load environment variables
load_dotenv()

# Initialize AgentOps
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
if not AGENTOPS_API_KEY:
    raise ValueError("AGENTOPS_API_KEY not found in environment variables")

agentops.init(AGENTOPS_API_KEY)

# Load configuration from environment variables with default values
CHAT_TITLE = os.getenv("CHAT_TITLE", "AI Chat Assistant")
WELCOME_MESSAGE = os.getenv("WELCOME_MESSAGE", "Welcome! How can I assist you today?")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant that answers questions based on the given context and chat history.")
ASTRA_COLLECTION_NAME = os.getenv("ASTRA_COLLECTION_NAME")

@agentops.record_function('create_app')
def create_app():
    app = FastAPI()
    
    templates_directory = os.path.join(os.path.dirname(__file__), "templates")
    templates = Jinja2Templates(directory=templates_directory)
    
    static_directory = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_directory), name="static")
    
    return app, templates

app, templates = create_app()

chat_history: List[Message] = []

project_root = os.path.dirname(os.path.abspath(__file__))

@agentops.record_function('initialize_rag_service')
def initialize_rag_service():
    chroma_db_path = os.path.join(project_root, "db")
    vector_store = AstraDBStore(collection_name=ASTRA_COLLECTION_NAME)
    return RAGService(vector_store)

rag_service = initialize_rag_service()

@app.get("/", response_class=HTMLResponse)
@agentops.record_function('read_root')
async def read_root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "chat_title": CHAT_TITLE,
            "welcome_message": WELCOME_MESSAGE,
        },
    )

@app.post("/chat")
@agentops.record_function('chat')
async def chat(request: Request, message: str = Form(...)) -> HTMLResponse:
    try:
        prepared_messages, citations = await rag_service.prepare_messages_with_sources(
            system_prompt=f"<system-prompt>{SYSTEM_PROMPT}</system-prompt>",
            chat_history=chat_history[-5:],
            user_message=message,
        )
        
        bot_response = await get_chat_response_with_history(prepared_messages)
        bot_response_html = markdown2.markdown(bot_response, safe_mode="escape")
        
        chat_history.append(Message(role=MessageRole.user, content=message))
        chat_history.append(Message(role=MessageRole.assistant, content=bot_response))
        
        message_id = str(uuid.uuid4())
        
        response_html = templates.TemplateResponse(
            "bot_message.html",
            {
                "request": request,
                "bot_response_html": bot_response_html,
                "citations": citations,
                "message_id": message_id,
            },
        )
        
        return response_html
    except Exception as e:
        agentops.log_error(f"Error in chat endpoint: {str(e)}")
        raise

@app.get("/api/chat_history")
@agentops.record_function('get_chat_history')
async def get_chat_history() -> List[Dict[str, str]]:
    return [message.model_dump() for message in chat_history]

@app.post("/api/clear_history")
@agentops.record_function('clear_history')
async def clear_history() -> Dict[str, str]:
    chat_history.clear()
    return {"message": "Chat history cleared"}

@agentops.record_function('main')
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        agentops.log_error(str(e))
    finally:
        agentops.end_session('Success')