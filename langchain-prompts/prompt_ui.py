
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

st.header("Research Tool")

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need","BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners","Diffusion Models Best GAN'S on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style",["Beginner Friendly","Technical","Code-Oriented","Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 Paragraphs)","Medium (4-5 Paragraphs)","Long (Detailed Explanation)"])

template = load_prompt('template.json')



if st.button("Summarize"):
    chain = template | chat_model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })

    content = result.content

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                st.markdown(block["text"])
            elif isinstance(block, str):
                st.markdown(block)
    else:
        st.markdown(content)
