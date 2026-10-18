from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

chat_model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)


parser = JsonOutputParser()

template = PromptTemplate(
  template = "Give me the name, age and city of a fictional person \n {format_instruction}",
  input_variable = [],
  partial_variables = {"format_instruction": parser.get_format_instructions()}
)

chain = template | chat_model | parser

final_result = chain.invoke({})

print(final_result)