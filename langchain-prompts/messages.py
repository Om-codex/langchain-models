from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# kind of chat history
messages = [
  SystemMessage(content = "You are a helpful assistant"),
  HumanMessage(content = "Tell me about LangChain in short")
]
result = chat_model.invoke(messages)
content = result.content
    
if isinstance(content, list):
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            final_result = block["text"]
        elif isinstance(block, str):
            final_result = block["text"]
        else:
            final_result = block["text"]

messages.append(AIMessage(content=final_result))

print(messages)