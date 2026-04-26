import os
from pypdf import PdfReader
from embed import embed_text, model
import faiss
from groq import Groq
import streamlit as st
api_key = st.secrets["GROQ_API_KEY"]

def load_docs(folder_path="data"):
    all_docs = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 🔹 TEXT FILES
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                docs = [line.strip() for line in f.read().split("\n") if line.strip()]
                all_docs.append({"source": filename, "content": docs})

        # 🔹 PDF FILES
        elif filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""

            for page in reader.pages:
                text += page.extract_text() + "\n"

            docs = [line.strip() for line in text.split("\n") if line.strip()]
            all_docs.append({"source": filename, "content": docs})

    return all_docs

def flatten_docs(docs):
    all_text = []

    for doc in docs:
        for line in doc["content"]:
            all_text.append({
                "source": doc["source"],
                "text": line
            })

    return all_text

def chunk_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks =[]
    for i in range(0,len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def chunk_docs(flat_docs, chunk_size=50, overlap=10):
    chunked_docs = []
    for doc in flat_docs:
        text = doc["text"]
        source = doc["source"]
        chunks = chunk_text(text, chunk_size, overlap)
        for chunk in chunks:
            chunked_docs.append({
                "source": source,
                "text": chunk
            })
    return chunked_docs

def extract_texts(chunked_docs):
    return [doc["text"] for doc in chunked_docs]

def create_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def retrieve(query, index, chunked_docs, model, k=5):
    import numpy as np

    # Convert query → embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')

    # Normalize
    faiss.normalize_L2(query_embedding)

    # Search
    D, I = index.search(query_embedding, k)

    results = []
    for i in I[0]:
        results.append(chunked_docs[i])

    return results

client = Groq(api_key=api_key)

def generate_answer(query, retrieved_docs):
    context = "\n\n".join([f"[Source: {doc['source']}]\n{doc['text']}" for doc in retrieved_docs])
    prompt = f"""
    You are a precise and reliable AI assistant.
    STRICT RULES:
    - Answer ONLY from the provided context
    - If answer is not in context, say "I don't know"
    - Do NOT guess or add extra information

    STYLE:
    - Be concise
    - Use bullet points when possible
    - Keep answers clear and structured

    Context:
    {context}

    Question:
    {query}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

def rag(query, index, chunked_docs, model):
    retrieved_docs = retrieve(query, index, chunked_docs, model, k=3)
    answer = generate_answer(query, retrieved_docs)
    return answer

def generate_stream(query, retrieved_docs):
    context = "\n".join([f"{doc['source']}: {doc['text']}" for doc in retrieved_docs])
    prompt = f"""
    You are a precise and reliable AI assistant.
    STRICT RULES:
    - Answer ONLY from the provided context
    - If answer is not in context, say "I don't know"
    - Do NOT guess or add extra information

    STYLE:
    - Be concise
    - Use bullet points when possible
    - Keep answers clear and structured

    Context:
    {context}

    Question:
    {query}
"""

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        stream=True
    )

    for chunk in stream:
        yield chunk.choices[0].delta.content or ""

def rag_stream(query, index, chunked_docs, model):
    retrieved_docs = retrieve(query, index, chunked_docs, model, k=3)
    return generate_stream(query, retrieved_docs)

class RAGPipeline:
    def __init__(self):
        from embed import embed_text, model

        print("🔄 Loading and processing documents...")

        self.docs = load_docs()
        self.flat_docs = flatten_docs(self.docs)
        self.chunked_docs = chunk_docs(self.flat_docs)

        self.texts = extract_texts(self.chunked_docs)
        self.embeddings = embed_text(self.texts)

        self.index = create_faiss_index(self.embeddings)
        self.model = model

        print("✅ Pipeline ready!")

    def query(self, query):
        retrieved_docs = retrieve(query, self.index, self.chunked_docs, self.model)
        return generate_answer(query, retrieved_docs)

    def stream(self, query):
        retrieved_docs = retrieve(query, self.index, self.chunked_docs, self.model)
        return generate_stream(query, retrieved_docs)

if __name__ == "__main__":
    rag = RAGPipeline()
    query = "How to be an AI expert?"
    print("\n🤖 Answer:\n")
    print(rag.query(query))

    print("\n⚡ Streaming:\n")
    for chunk in rag.stream(query):
        print(chunk, end="", flush=True)