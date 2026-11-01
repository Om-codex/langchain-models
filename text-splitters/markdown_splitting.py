from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
# My Heading

This is **bold** and this is _italic_ text.

- First bullet point
- Second bullet point

> A blockquote goes here.

[Link to Brave](https://brave.com)
```It looks like my previous reply actually finished completely — the markdown block was closed properly, so nothing was truncated.

If you'd like, I can add more elements (tables, code blocks, task lists) to the snippet. Just say the word!
"""
splitter = RecursiveCharacterTextSplitter.from_language(
  language = Language.MARKDOWN,
  chunk_size = 250,
  chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])