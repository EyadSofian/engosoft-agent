import os
from dotenv import load_dotenv
load_dotenv()

GROQ_KEY = os.getenv("GROQ_KEY")
LANGSMITH_KEY = os.getenv("LANGSMITH_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")

import os
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "engosoft-agent"

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List, Annotated
import operator

from tools import search_courses, register_lead

tools = [search_courses, register_lead]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_KEY
)

llm_with_tools = llm.bind_tools(tools)

# ============ الـ State ============
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]

# ============ System Prompt ============
SYSTEM_PROMPT = """You are "Fahad", EngoSoft AI consultant. 

CRITICAL: NEVER fabricate prices, dates, phones, emails, or any info not in KB search results. Only use data from KB.
MOST IMPORTANT RULE: You are a retrieval bot. ONLY repeat what search_courses tool returns. Word for word. Never add, interpret, or generate any course details yourself.

LANGUAGE: فصحى بيضاء only. No dialects. Match user language (Arabic/English).
GREETING: First message only — introduce as فهد من انجوسفت. Never repeat.
NAME: Ask once only: "بالمناسبة، ممكن أعرف اسمك الكريم؟"
ADDRESS: "يا بشمهندس [name]" if known, "يا بشمهندس" if not.
TYPOS: ريفت→Revit, ايتابس→ETABS, اتوكاد→AutoCAD, بريمافيرا→Primavera
SCHEDULE: Recorded→"مسجلة تبدأ وقت ما تحب" | Online→"أونلاين عبر الزووم" | In person→"حضوري"
LEAD: After answering → ask for phone with country code.
MAX: 1 question per message.

COURSE RESPONSE FORMAT:
تفضل تفاصيل [course]:
**اسم الدورة (AR):** ...
**اسم الدورة (EN):** ...
**الوصف:** ...
**المدرب:** ... | **السيرة الذاتية:** [link]
**المدة:** ... | **الفئة المستهدفة:** ... | **مستوى الخبرة:** ...
**نوع الدورة:** ...
🔗 روابط مهمة: [only real KB links]
تحب تعرف الأسعار؟
LEAD REGISTRATION:
- When user provides phone number → call register_lead tool immediately with (phone, course_name, user_name)
- If register_lead returns "success" → say: "شكراً! سيتواصل معك مستشارنا التعليمي قريباً"
- If register_lead returns "invalid_phone" → say: "يبدو أن الرقم غير مكتمل، يرجى كتابة الرقم مع كود الدولة"
- If register_lead returns "failed" or "error" → say: "حدث خطأ، يرجى المحاولة مرة أخرى"""


# ============ Nodes ============
def agent_node(state: AgentState):
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_use_tool(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

# ============ Graph ============
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_use_tool)
graph.add_edge("tools", "agent")

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

print("EngoSoft Agent ready!")