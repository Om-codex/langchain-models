from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

parser = StrOutputParser()

chain = template1 | chat_model | parser | template2 | chat_model | parser

result = chain.invoke({"topic": "black hole"})

print(result)