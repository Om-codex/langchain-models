from langchain_text_splitters import  RecursiveCharacterTextSplitter

text = """
Large language models operate by predicting the next token in a sequence based on vast patterns learned during training, effectively creating a statistical mirror of human text rather than possessing true understanding or consciousness. While they can generate code, write poetry, and answer complex queries with surprising fluency, their outputs are fundamentally rooted in probability distributions derived from their training data, which means they can sometimes hallucinate facts with complete confidence. The ongoing evolution of these systems involves not just scaling up parameter counts, but also refining alignment techniques to ensure they remain helpful, harmless, and honest, bridging the gap between raw computational power and nuanced, context-aware interaction.
"""

#splitter
splitter = RecursiveCharacterTextSplitter(
  chunk_size = 300,
  chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)