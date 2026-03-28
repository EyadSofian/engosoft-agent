import os
import requests
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.tools import tool

embeddings = SentenceTransformerEmbeddings(
    model_name="paraphrase-multilingual-mpnet-base-v2"
)

vectordb = Chroma(
    persist_directory="./vectordb",
    embedding_function=embeddings
)

@tool
def search_courses(query: str) -> str:
    """Search EngoSoft courses database. Use this for any question about courses. IMPORTANT: Use ONLY the data returned here. NEVER add or modify any information."""
    results = vectordb.similarity_search(query, k=5)
    if not results:
        return "No courses found. Tell user you cannot find this course."
    
    output = "USE ONLY THIS DATA - DO NOT ADD ANYTHING:\n\n"
    for i, r in enumerate(results, 1):
        output += f"=== RESULT {i} ===\n{r.page_content}\n\n"
    output += "\nCRITICAL: Only use the exact data above. Never add prices, dates, phones, or any other info."
    return output

@tool
def register_lead(phone: str, course_name: str, user_name: str = "") -> str:
    """Register a potential student lead when they provide their phone number. Call this when user gives their phone number."""
    digits = ''.join(filter(str.isdigit, phone))
    if len(digits) < 7 or len(digits) > 15:
        return "invalid_phone"
    
    payload = {
        "name": user_name if user_name else "مجهول",
        "phone": phone,
        "course": course_name,
        "source": "LangGraph Agent"
    }
    
    try:
        response = requests.post(
            "https://n8n.engosoft.com/webhook/engosoft-courses",
            json=payload,
            timeout=10
        )
        if response.status_code == 200:
            return "success"
        else:
            return "failed"
    except Exception as e:
        return f"error: {str(e)}"