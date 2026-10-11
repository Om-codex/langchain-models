from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model="gpt-5k-mini")

result = chat_model.invoke("What is the capital of India?")

print(result.content)