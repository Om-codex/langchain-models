from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings(model = "text-embedding-3-large",dimensions=32)

documents = ["Delhi is the capital of India",
             "Kolkata is the capital of WestBengal",
             "Paris is the capital of France."]

result = embedding_model.embed_documents(documents)

print(str(result))