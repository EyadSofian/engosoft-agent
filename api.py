from dotenv import load_dotenv
load_dotenv()

GROQ_KEY = os.getenv("GROQ_KEY")
LANGSMITH_KEY = os.getenv("LANGSMITH_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")

import os
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "engosoft-agent"

from fastapi import FastAPI
from pydantic import BaseModel
from agent import app as agent_app
from langchain_core.messages import HumanMessage

api = FastAPI()

class Message(BaseModel):
    message: str
    session_id: str = "default"

@api.get("/")
def health():
    return {"status": "ok", "agent": "Fahad - EngoSoft"}

@api.post("/chat")
def chat(msg: Message):
    config = {"configurable": {"thread_id": msg.session_id}}
    result = agent_app.invoke(
        {"messages": [HumanMessage(content=msg.message)]},
        config=config
    )
    return {"response": result["messages"][-1].content}