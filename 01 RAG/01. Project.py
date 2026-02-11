"""
Project 1: RAG with Semantic Search (OpenAI Embeddings)

"""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = InferenceClient(token=os.getenv("HUGGING_FACE_API_KEY"))

def load_knowledge_base(knowledge_dir: str = "knowledge") -> dict:
    """
    STEP 1: PREPARE DOCUMENTS - Load from external files
    """
    print("\n" + "="*70)
    print("STEP 1: Loading documents from files")
    print("="*70)

    knowledge_base = {}
    knowledge_path = Path(__file__).parent / knowledge_dir

    if not knowledge_path.exists():
        print(f'Error: knowledge directory not found: {knowledge_path}')
        return knowledge_base
    
    for file_path in knowledge_path.glob('*.md'):
        doc_id = file_path.stem
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            knowledge_base[doc_id] = content
        
        print(f"Loaded {doc_id} ({len(content)} characters)")

    return knowledge_base

def get_embedding(text: str, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> list:
    text = text.replace("\n", " ")
    response = client.feature_extraction(text, model=model)
    return response

def cosine_similarity(vec1: list, vec2: list) -> float:

    a = np.array(vec1)
    b = np.array(vec2)

    dot_product = np.dot(a, b)

    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    
    return dot_product / (magnitude_a * magnitude_b)

def create_embeddings_cache(knowledge_base: dict) -> dict:

    print("\n" + "="*70)
    print("STEP 2: Creating embeddings for all documents")
    print("="*70)
    print("Converting each document to a n-dimensional vector...\n")

    embeddings_cache = {}

    for doc_id, content in knowledge_base.items():
        print(f" Creating embedding for: {doc_id} ...")
        embedding = get_embedding(content)
        embeddings_cache[doc_id] = embedding

    print(f"\n Created {(len(embeddings_cache))} embeddings (cache in memory)")

    return embeddings_cache

def semantic_search(query: str, knowledge_base: dict, embeddings_cache: dict, top_k: int = 2) -> list:

    print("\n" + "="*70)
    print("STEP 3: Semantic search with cosine similarity")
    print("="*70)

    query_embedding = get_embedding(query)

    similarities = []
    print(f"Calculating cosine similarity for each document: ")

    for doc_id, content in knowledge_base.items():
        doc_embedding = embeddings_cache[doc_id]
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((doc_id, content, similarity))
        print(f" -{doc_id}: {similarity: .4f}")

    similarities.sort(key=lambda x: x[2], reverse=True)
    print(f" \n Top {top_k} most relevant documents selected")

    return similarities[:top_k]

def rag_query(user_question, knowledge_base, embeddings_cache):

    # Step 1: Search for relevant documents
    relevant_docs = semantic_search(user_question, knowledge_base, embeddings_cache)

    if not relevant_docs:
        return "I don't have information about that in the knowledgebase"

    # Step 2: Build context from retrieved documents
    context_parts = []
    for doc_id, content, similarity in relevant_docs:
        context_parts.append(f"[Document: {doc_id}]\n{content}")

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Context augmentation
    prompt = f""" Answer the question based only on provided documents.

    context:
    {context}

    question: {user_question} 
    
    Answer:"""

    # Step 4: Generate answer
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=[
            {"role": "system", "content": "You are helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content

def main():
    """
    Run example RAG queries with semantic search
    """
    print("=" * 70)
    print("RAG Assistant - Semantic Search with Embeddings")
    print("=" * 70)

    # Step 1: Load documents
    knowledge_base = load_knowledge_base("knowledge")

    # Step 2: Creating embedding
    embeddings_cache = create_embeddings_cache(knowledge_base)

    # Ask question
    questions = [
        "What is remote work policy",
        "How much does Nexus Cloud Platform cost?",
        "What health benefits do employees get?"
    ]

    for question in questions:
        answer = rag_query(question, knowledge_base, embeddings_cache)
        print(f"\n{'='*70}")
        print(f"ANSWER: {answer}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
