from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


# 1st prompt
template1 = PromptTemplate(
  template = "Write a detailed report on {topic}",
  input_variables = ["topic"]
)

# 2nd prompt
template2 = PromptTemplate(
  template = "Write a 5 line summary on the following text. /n {text}",
  input_variables = ["text"]
)

prompt1 = template1.invoke({"topic" : "black hole"})

result = chat_model.invoke(prompt1)

prompt2 = template2.invoke({"text":result.content})

result1 = chat_model(prompt2)

print(result1.content)