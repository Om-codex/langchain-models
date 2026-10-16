import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# Suppress the Google GenAI SDK automatic function calling warning
warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

chat_history = [
    SystemMessage(content="You are a helpful Research AI Assistant")
]

while True:
    user_input = input("You:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = chat_model.invoke(chat_history)
    content = result.content
    
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                final_result = block["text"]
            elif isinstance(block, str):
                final_result = block["text"]
            else:
                final_result = block["text"]

    chat_history.append(AIMessage(content=final_result))
    print("AI: ",final_result)

print(chat_history)