from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

chat_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)


parser = JsonOutputParser()

template = PromptTemplate(
  template = "Give me the name, age and city of a fictional person \n {format_instruction}",
  input_variable = [],
  partial_variables = {"format_instruction": parser.get_format_instructions()}
)

prompt = template.format()

result = chat_model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
print(type(final_result))