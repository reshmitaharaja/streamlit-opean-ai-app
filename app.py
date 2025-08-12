import os
import streamlit as st
import pickle
import time
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Streamlit UI
st.title("📰 RockyBot: News Research Tool")
st.sidebar.title("Enter News Article URLs")

# Get URLs from user
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

# Create LLM
llm = OpenAI(temperature=0.9, max_tokens=500)

# When Process button clicked
if process_url_clicked:
    if urls:
        loader = UnstructuredURLLoader(urls=urls)
        st.info("📥 Loading data...")
        data = loader.load()

        # Split into chunks
        st.info("✂ Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(data)

        # Create embeddings and store
        st.info("🧠 Creating embeddings...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        # Save to disk
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)
        st.success("✅ Data processed successfully!")

# Ask question
query = st.text_input("Ask me anything about the articles:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
        result = chain({"question": query}, return_only_outputs=True)

        st.subheader("Answer:")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("Sources:")
            st.write(result["sources"])
