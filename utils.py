from langchain.embeddings import HuggingFaceEmbeddings


def load_embeddings():
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return hf_embeddings