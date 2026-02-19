from pathlib import Path
import faiss
import numpy as np

# ---------------------------
# 1. Setup paths using pathlib
# ---------------------------
index_dir = Path("faiss_index")
index_dir.mkdir(exist_ok=True)               # create folder if it doesn't exist
index_file = index_dir / "index.faiss"      # full path to index file

# ---------------------------
# 2. Prepare your data
# ---------------------------
d = 128                                      # dimension of vectors
num_vectors = 1000                           # number of vectors

# Random example vectors (replace this with your real embeddings)
xb = np.random.random((num_vectors, d)).astype('float32')

# ---------------------------
# 3. Create or load FAISS index
# ---------------------------
if index_file.exists():
    print(f"Loading existing index from {index_file}")
    index = faiss.read_index(str(index_file))
else:
    print("Creating a new FAISS index...")
    index = faiss.IndexFlatL2(d)            # L2 distance index
    index.add(xb)                            # add vectors to index
    faiss.write_index(index, str(index_file))
    print(f"Index saved to {index_file}")

# ---------------------------
# 4. Query the index
# ---------------------------
# Example query vector
query = np.random.random((1, d)).astype('float32')
k = 5  # number of nearest neighbors to search

distances, indices = index.search(query, k)
print("Nearest neighbor indices:", indices)
print("Distances:", distances)