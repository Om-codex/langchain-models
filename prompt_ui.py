
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

st.header("Research Tool")

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need","BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners","Diffusion Models Best GAN'S on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style",["Beginner Friendly","Technical","Code-Oriented","Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 Paragraphs)","Medium (4-5 Paragraphs)","Long (Detailed Explanation)"])

#template
template = PromptTemplate(
    template="""
You are an expert AI research assistant specializing in analyzing academic research papers and explaining complex technical concepts clearly and accurately.

## Research Paper
Paper name: {paper_input}

## User Preferences
- Explanation style: {style_input}
- Explanation length: {length_input}

## Instructions

Analyze and explain the specified research paper according to the user's preferences.

1. **Research context:** Explain the problem the paper addresses, why it matters, and the limitations of previous approaches.

2. **Core idea:** Describe the paper's main contribution, proposed methodology, and key innovations.

3. **Technical explanation:** Explain important architectures, algorithms, mathematical concepts, experiments, and evaluation metrics relevant to the paper.

4. **Examples:** Use relevant examples, analogies, or practical applications according to the selected explanation style.

5. **Results and findings:** Summarize the experiments, reported results, comparisons with baselines, and conclusions. Do not invent numerical results.

6. **Limitations:** Discuss the paper's limitations, assumptions, trade-offs, and possible future improvements when supported by the paper.

7. **Impact:** Explain the paper's significance and influence on subsequent research or real-world applications, where supported by reliable evidence.

## Response Rules

- Follow the selected explanation style and length throughout the entire response.
- Maintain a consistent structure using headings, subheadings, bullet points, and tables where useful.
- Define technical terms before using them extensively.
- Preserve scientific accuracy and distinguish the authors' claims from your own interpretation.
- Do not fabricate citations, experimental results, equations, or claims.
- If the exact paper cannot be identified or its contents are unavailable, state this clearly instead of pretending to have read it.
- Avoid unnecessary repetition and filler.

Generate a coherent, research-focused explanation tailored to the user's selected preferences.
""",
input_variables = ['paper_input','style_input','length_input']
)

prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})

if st.button("Summarize"):
    result = chat_model.invoke(prompt)

    content = result.content

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                st.markdown(block["text"])
            elif isinstance(block, str):
                st.markdown(block)
    else:
        st.markdown(content)
