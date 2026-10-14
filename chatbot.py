import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Suppress the Google GenAI SDK automatic function calling warning
warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

while True:
    user_input = input("You:")
    if user_input == 'exit':
        break
    result = chat_model.invoke(user_input)
    content = result.content
    
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                print("AI: ", block["text"])
            elif isinstance(block, str):
                print("AI: ", block)
            else:
                print("AI: ", content)
    else:
        print("AI: ", content)