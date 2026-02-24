import faiss
import numpy as np
import pickle

INDEX_FILE = "vector_store.index"
DATA_FILE = "text_data.pkl"

def save_to_vector_store(embeddings, texts):
    # Convert list of embeddings to a float32 numpy array
    dimension = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')
    
    # Create the FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    # Save the index and the actual Arabic text
    faiss.write_index(index, INDEX_FILE)
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(texts, f)

def search_vector_store(query_embedding, top_k=3):
    index = faiss.read_index(INDEX_FILE)
    with open(DATA_FILE, 'rb') as f:
        texts = pickle.load(f)
    
    query_np = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_np, top_k)
    
    # Return the text chunks corresponding to the best indices
    return [texts[i] for i in indices[0] if i != -1]