# -------------------- PDF Processing --------------------
from PyPDF2 import PdfReader

# -------------------- LangChain Imports (Modern API) --------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# -------------------- Gemini Model --------------------
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------- Streamlit --------------------
import streamlit as st

# -------------------- ENV FILES --------------------
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env


# -------------------- Load HuggingFace Embeddings --------------------
from utils.helper import load_embeddings 

hf_embeddings = load_embeddings()


# -------------------- Extract Text from PDFs --------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text


# -------------------- Split Text into Chunks --------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)


# -------------------- Create FAISS Vector Store --------------------
def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=hf_embeddings
    )
    vector_store.save_local("faiss_index")


# -------------------- Create Modern Retrieval Chain --------------------
def get_conversational_chain(retriever):

    # Prompt Template (Modern Style)
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

    # Gemini Model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    # Create document chain
    document_chain = create_stuff_documents_chain(model, prompt)

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


# -------------------- Handle User Question --------------------
def user_input(user_question):
    # Load FAISS index
    new_db = FAISS.load_local(
        "faiss_index",
        hf_embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = new_db.as_retriever()

    chain = get_conversational_chain(retriever)

    # Invoke modern chain
    response = chain.invoke({"input": user_question})

    st.write("Reply:")
    st.write(response["answer"])


# -------------------- Streamlit App --------------------
def main():
    st.set_page_config(page_title="Chat PDF", layout="centered")
    st.header("Chat with PDF using Gemini 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click Submit & Process",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done Processing!")


if __name__ == "__main__":
    main()
