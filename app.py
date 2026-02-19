# -------------------- Standard Imports --------------------
from pathlib import Path
from PyPDF2 import PdfReader
import streamlit as st
from dotenv import load_dotenv

# -------------------- LangChain Imports (Modern API) --------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# -------------------- Gemini Model --------------------
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------- Load ENV --------------------
load_dotenv()  # Load environment variables

# -------------------- Load HuggingFace Embeddings --------------------
from utils.helper import load_embeddings
hf_embeddings = load_embeddings()


# -------------------- PDF Text Extraction --------------------
def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


# -------------------- Text Chunking --------------------
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)


# -------------------- FAISS Vector Store --------------------
def get_vector_store(text_chunks, embeddings, index_dir="faiss_index", index_name="index"):
    """
    Creates or loads a FAISS vector store.
    If index exists, loads it; otherwise creates a new one from text chunks.
    """
    index_path = Path(index_dir)
    index_path.mkdir(exist_ok=True)
    pkl_file = index_path / f"{index_name}.pkl"

    if pkl_file.exists():
        st.info("Loading existing FAISS index...")
        vector_store = FAISS.load_local(index_path, embeddings, index_name=index_name)
    else:
        st.info("Creating new FAISS index...")
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        vector_store.save_local(index_path, index_name=index_name)
        st.success(f"FAISS index saved at {index_path}")
    
    return vector_store


# -------------------- Conversational Chain --------------------
def get_conversational_chain(retriever):
    """Creates a modern LangChain retrieval + LLM chain."""
    prompt = ChatPromptTemplate.from_template("""
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {input}

    Answer:
    """)

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


# -------------------- Handle User Question --------------------
def user_input(user_question):
    """Handles user query and returns response from FAISS + Gemini."""
    index_path = Path("faiss_index")
    vector_store = FAISS.load_local(
        index_path,
        hf_embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever()
    chain = get_conversational_chain(retriever)

    response = chain.invoke({"input": user_question})
    st.write("Reply:")
    st.write(response["answer"])


# -------------------- Streamlit App --------------------
def main():
    st.set_page_config(page_title="Chat PDF", layout="centered")
    st.header("Chat with PDF using Gemini 💁")

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, hf_embeddings)
                    st.success("Processing complete! FAISS index is ready.")
            else:
                st.warning("Please upload at least one PDF file.")

    # User query input
    user_question = st.text_input("Ask a question from the PDFs")
    if user_question:
        if Path("faiss_index").exists():
            user_input(user_question)
        else:
            st.warning("Please upload and process PDFs first to create FAISS index.")


if __name__ == "__main__":
    main()
