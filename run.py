import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

from agent import app
from langchain_core.messages import HumanMessage

print("=" * 50)
print("EngoSoft AI Agent - Fahad")
print("=" * 50)
print("Type 'exit' to quit\n")

thread_id = "user_session_1"
config = {"configurable": {"thread_id": thread_id}}

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    if not user_input.strip():
        continue
    
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    
    last_message = result["messages"][-1]
    print(f"\nFahad: {last_message.content}\n")
    print("-" * 50)