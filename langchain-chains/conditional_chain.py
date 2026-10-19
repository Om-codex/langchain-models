from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# Groq AI Model
model = ChatGroq(
  model = "openai/gpt-oss-20b"
)

prompt1 = PromptTemplate(
  template = "Classify the sentiment of the following feedback text into positive or negative \n{feedback}",
  input_variables = ['feedback']
)
prompt2 = PromptTemplate(
  template = "Generate 5 short quiz question and answer from the following text \n {text}",
  input_variables = ['text']
)

parser = StrOutputParser()

classifier_chain = prompt1 | model | parser
result = classifier_chain.invoke({"feedback":"This new smartphone has cool features I love it!"})
print(result)