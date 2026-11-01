from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from load.

embedding = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")

splitter = SemanticChunker(
  HuggingFaceEmbeddings(), breakpoint_threshold_type = "standard_deviation",
  breakpoint_threshold_amount = 1
)

sample = """
Farmers were working hard in the feilds, preparing the soil fo the next season.The sun was bright, and the air smelled of earth and fresh greass.The Indian Premiere Leagure {IPL} is the biggest cricket league in the world.People all over the world.People all over the world with the matches and cheer for their favourite teams.
"""