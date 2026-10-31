from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="content_refresh_anonymized.csv")

docs = loader.load()

print(docs[0])