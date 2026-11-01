from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

documents = [
  Document(page_content ="Langchain helps developers build LLM applications easily."),
  Document(page_content ="Chroma is a vector database optimized for LLM-based search."),
  Document(page_content ="Embeddings convert text into high-dimensional vectors."),
  Document(page_content ="OpenAI provides powerful embedding models.")
]


embedding_model = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5"
)

vector_store = Chroma.from_documents(
  documents = documents,
  embedding = embedding_model,
  collection_name = "my_collection"
)

# code to create a retriever
retriever = vector_store.as_retriever(search_kwargs = {"k":2})

query = "What is chroma used for?"
results = retriever.invoke(query)

print(results)