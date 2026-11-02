import os
from google.colab import userdata
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_classic.retrievers import MultiQueryRetriever

# Securely set the GROQ_API_KEY environment variable from Colab secrets
try:
    os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
except Exception as e:
    print("Warning: Could not load GROQ_API_KEY from Colab secrets. Please ensure it is set under the Secrets panel (🔑).")

embedding_model = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5"
)

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

vector_store = FAISS.from_documents(
    documents = all_docs,
    embedding = embedding_model
)

similarity_retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs ={"k":5})

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever = vector_store.as_retriever(search_kwargs = {"k":5}),
    llm = ChatGroq(model = "openai/gpt-oss-20b")
)

query = "How to improve energy levels and maintain balance?"

similarity_results = similarity_retriever.invoke(query)
multi_query_results = multi_query_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
  print(f"----Similarity Result {i+1}----")
  print(doc.page_content)

print("="*150)

for i, doc in enumerate(multi_query_results):
  print(f"----Multi Query Retriever Result {i+1}----")
  print(doc.page_content)