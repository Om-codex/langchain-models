# Method 1
from langchain_community.document_loaders import TextLoader

loader = TextLoader("text.txt", encoding='utf-8')

docs = loader.load()
print(docs)
print(type(docs))

# Method 2 langchain independent
from pathlib import Path
from langchain_core.documents import Document

file_path = Path("text.txt")

text = file_path.read_text(encoding="utf-8")

document = Document(
    page_content=text,
    metadata={"source": str(file_path)}
)

print(document.page_content)
print(document.metadata)
print(type(document))