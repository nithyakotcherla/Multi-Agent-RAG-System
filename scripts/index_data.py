import json
import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from scripts.cost_tracker import track_cost

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Load from corpus.json
def load_corpus(filepath="data/corpus.json"):
    with open(filepath) as f:
        return json.load(f)

# Step 2: Chunk with metadata
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def chunk_corpus(corpus, chunk_size=500, overlap=100):
    chunks = []
    for entry in corpus:
        text_chunks = chunk_text(entry["text"], chunk_size, overlap)
        for chunk in text_chunks:
            chunks.append({
                "text": chunk,
                "page": entry["page"],
                "source": entry["source"]
            })
    return chunks

# Step 3: Embed chunks
def get_embeddings(texts):
    if not texts:
        return []
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    track_cost(response, is_embedding=True)
    return [item.embedding for item in response.data]

def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = []
    # Batch process in sizes of 50
    for i in range(0, len(texts), 50):
        print(f"Embedding chunks {i} to {min(i+50, len(texts))}...")
        embeddings.extend(get_embeddings(texts[i:i+50]))
    return embeddings

# Step 4: Build and save FAISS index
def main():
    print("Loading corpus...")
    corpus = load_corpus()
    
    print("Chunking corpus...")
    chunks = chunk_corpus(corpus)
    print(f"Total chunks: {len(chunks)}")
    
    print("Generating embeddings (this may take a minute)...")
    embeddings = embed_chunks(chunks)
    vectors = np.array(embeddings).astype('float32')
    
    print("Building FAISS index...")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(vectors)
    index.add(vectors) # type: ignore
    
    print("Saving index and chunks...")
    faiss.write_index(index, "data/my_index.faiss")
    with open("data/chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()
