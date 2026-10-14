from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions = 300)

documents = [
    "Virat Kohli is widely regarded as one of the greatest batsmen in the history of cricket, known for his aggressive run-scoring and exceptional fitness.",
    "MS Dhoni captained the Indian cricket team to victory in the 2011 ICC Cricket World Cup and is celebrated for his calm leadership and finishing abilities.",
    "Sachin Tendulkar holds the record for the most runs in international cricket and is often called the 'God of Cricket' in India.",
    "Rohit Sharma has scored three double centuries in One Day Internationals, more than any other player, and leads the Indian team across formats.",
    "Jasprit Bumrah is recognized as one of the best bowlers in wld cricket, famous for his unique action and lethal yorkers."
]

query = "tell me about virat kohli"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

print(cosine_similarity([query_embedding], doc_embeddings))