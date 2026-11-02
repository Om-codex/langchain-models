from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding_model = HuggingFaceEmbeddings(
  model_name = "BAAI/bge-small-en-v1.5"
)
documents = [
    # LangChain
    Document(page_content="LangChain is a framework for developing applications powered by large language models."),
    Document(page_content="LangChain provides tools for building chatbots, AI agents, and retrieval augmented generation systems."),
    Document(page_content="LangChain simplifies connecting LLMs with external tools, APIs, and databases."),
    Document(page_content="Developers use LangChain to create context-aware AI applications."),

    # Vector Databases
    Document(page_content="ChromaDB is a vector database used to store, index, and retrieve text embeddings."),
    Document(page_content="FAISS enables efficient similarity search over high-dimensional vector representations."),
    Document(page_content="Pinecone is a managed vector database designed for scalable similarity search."),
    Document(page_content="Vector databases store embeddings and retrieve documents based on semantic similarity."),

    # Embeddings
    Document(page_content="Text embeddings convert sentences and paragraphs into numerical vector representations."),
    Document(page_content="Embedding models capture the semantic meaning of text in a high-dimensional vector space."),
    Document(page_content="Similar sentences generally have embeddings that are close together in vector space."),

    # RAG
    Document(page_content="Retrieval Augmented Generation combines document retrieval with large language model generation."),
    Document(page_content="RAG systems retrieve relevant information from external knowledge sources before generating answers."),
    Document(page_content="A typical RAG pipeline consists of document loading, text splitting, embedding, retrieval, and generation."),

    # Retrieval Techniques
    Document(page_content="Similarity search retrieves documents that are closest to a query in embedding space."),
    Document(page_content="MMR retrieval balances query relevance with diversity among the retrieved documents."),
    Document(page_content="Maximum Marginal Relevance reduces redundant results by selecting documents that offer different information."),
    Document(page_content="A similarity retriever may return several nearly identical documents when their embeddings are very close."),

    # LLMs
    Document(page_content="Large language models generate human-like text using patterns learned during training."),
    Document(page_content="GPT models are autoregressive language models that predict the next token in a sequence."),
    Document(page_content="Fine-tuning adapts a pretrained language model to perform a specific task.")
]

vector_store = Faiss.from_documents(
  documents = documents,
  embedding = embedding_model
)
retriever = vector_store.as_retriever(
  search_type = "mmr",
  search_kwargs = {"k":3, "lambda_mult":1}
)

query = "What is langchain?"
resutls = retriever.invoke(query)

for i, doc in results:
  print(f"\n----Result {i+1}----")
  print(doc.page_content)