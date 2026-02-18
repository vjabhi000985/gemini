from langchain_community.embeddings import HuggingFaceEmbeddings


def load_embeddings():
    """
    Load HuggingFace embedding model from local models folder.
    """

    model_path = "models/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},  # Change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings