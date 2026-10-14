
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

st.header("Research Tool")

user_input = st.text_input("Enter your prompt")

if st.button("Summarize"):
    result = chat_model.invoke(user_input)

    content = result.content

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                st.markdown(block["text"])
            elif isinstance(block, str):
                st.markdown(block)
    else:
        st.markdown(content)
