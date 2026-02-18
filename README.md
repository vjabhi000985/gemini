# Gemini RAG Pipeline

A simple Retrieval-Augmented Generation (RAG) application built with:

- Streamlit
- LangChain (0.2.x)
- Google Gemini
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- FAISS

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd gemini
```

### 2. Install dependencies (using uv)
```bash
uv venv
uv sync
```

### 3. Add Environment Variables
- Create a `.env` file in the project root:

```bash 
GOOGLE_API_KEY=your_api_key_here
```

### 4. Download Embedding Model

#### If not already available:
```bash
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('models/all-MiniLM-L6-v2')"
```

### 5. Run the App
```bash
uv run streamlit run app.py
```

### 6. Open in browser:
```bash
http://localhost:8501
```

### Note:
- `.env` is ignored for security.
- Embedding model can be tracked using Git LFS.
- Recommended Gemini model: gemini-1.5-pro.