from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
  model = "openai/gpt-oss-20b"
)
prompt = PromptTemplate(
  template = "Answer the following question \n{question} from the following text - \n{text}",
  input_variables = ['question','text']
)

parser = StrOutputParser()

url = "https://www.flipkart.com/apple-macbook-pro-m5-max-2026-m5-max-36-gb-2-tb-ssd-tahoe-mged4hn-a/p/itmff753615fba4e?pid=COMHZQX4M9Y7DQYP&lid=LSTCOMHZQX4M9Y7DQYPDP0PT6&hl_lid=&marketplace=FLIPKART&fm=eyJ3dHAiOiJyZWNvIiwicHJwdCI6InBwIiwibWlkIjoicHJvZHVjdFJlY29tbWVuZGF0aW9uL2FzcGVjdFNpbWlsYXIifQ%3D%3D&pageUID=1793447590127"

loader = WebBaseLoader(url)
docs = loader.load()
chain = prompt | model | parser
print(chain.invoke({"question":"What is the storage of this product?", "text":docs[0].page_content}))