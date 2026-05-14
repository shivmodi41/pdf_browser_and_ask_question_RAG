import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load Environment Variables
load_dotenv()

# Get GROQ API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check API Key
if not GROQ_API_KEY:
    st.error("GROQ API Key not found.")
    st.info('Create a .env file and add:\nGROQ_API_KEY="your_api_key"')
    st.stop()

# Streamlit Page
st.set_page_config(page_title="PDF Chatbot")

st.title("📄 PDF Chatbot using Groq AI")

# Sidebar
with st.sidebar:
    st.header("Upload PDF")
    file = st.file_uploader(
        "Upload your PDF file",
        type="pdf"
    )

# Process PDF
if file is not None:

    with st.spinner("Reading PDF..."):

        try:
            # Read PDF
            pdf_reader = PdfReader(file)

            text = ""

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text

            # Validate Text
            if not text.strip():
                st.error("No readable text found in PDF.")
                st.stop()

            # Split Text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_text(text)

            # Validate Chunks
            if not chunks:
                st.error("No text chunks created.")
                st.stop()

            # Create Embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Create Chroma Vector Store
            vector_store = Chroma.from_texts(
                texts=chunks,
                embedding=embeddings
            )

            st.success("PDF Processed Successfully!")

        except Exception as e:
            st.error(f"PDF Processing Error: {e}")
            st.stop()

    # Chat Input
    user_question = st.chat_input(
        "Ask a question from the PDF"
    )

    # If User Asks Question
    if user_question:

        with st.spinner("Generating Answer..."):

            try:
                # Similarity Search
                docs = vector_store.similarity_search(
                    user_question,
                    k=3
                )

                # Create Context
                context = "\n".join(
                    [doc.page_content for doc in docs]
                )

                # Initialize LLM
                llm = ChatGroq(
                    model_name="llama-3.1-8b-instant",
                    api_key=GROQ_API_KEY,
                    temperature=0.1,
                    max_tokens=500
                )

                # Prompt
                prompt = f"""
                You are a helpful AI assistant.

                Answer the question only using the provided context.

                Context:
                {context}

                Question:
                {user_question}
                """

                # Generate Response
                response = llm.invoke(prompt)

                # Display Chat
                st.chat_message("user").write(user_question)
                st.chat_message("assistant").write(response.content)

            except Exception as e:
                st.error(f"Answer Generation Error: {e}")