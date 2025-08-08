import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up Streamlit UI
st.set_page_config(page_title="Excel Q&A Assistant", layout="wide")
st.title("AI Agent for QA on Shared Excel Documents")

# Load Excel Files
@st.cache_data
def load_combined_text():
    file_paths = ["data.xlsx", "Forcast.xlsx"]
    combined_text = ""
    for file in file_paths:
        df = pd.read_excel(file)
        for row in df.itertuples(index=False):
            combined_text += " | ".join([str(cell) for cell in row if pd.notna(cell)]) + "\n"
    return combined_text

#Convert text into LangChain Documents
def convert_to_documents(text):
    return [Document(page_content=text)]

# Process & Create Vector DB
with st.spinner("Loading and processing documents..."):
    combined_text = load_combined_text()
    docs = convert_to_documents(combined_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)


    filtered_splits = [doc for doc in splits if len(doc.page_content) < 3000]
    embeddings = HuggingFaceEmbeddings(model="Qwen3-Embedding-0.6B")
    vectorstore = Chroma.from_documents(documents=filtered_splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

st.success("Document Processing Complete")

# Question Answering Interface
st.subheader("Ask your question")
user_question = st.text_input("Enter your question here:")

if user_question:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(user_question)
        st.markdown(f"**Answer:** {answer}")
