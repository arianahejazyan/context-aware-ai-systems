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

    
    return dot_product / (magnitude_a * magnitude_b)



def create_embeddings_cache(knowledge_base: dict) -> dict:

    print("\n" + "="*70)
    print("STEP 2: Creating embeddings for all documents")
    print("="*70)
    print("Converting each document to a 1536-dimensional vector...\n")


    return embeddings_cache


def semantic_search(query: str, knowledge_base: dict, embeddings_cache: dict, top_k: int = 2) -> list:

    print("\n" + "="*70)
    print("STEP 3: Semantic search with cosine similarity")
    print("="*70)
    
    return similarities[:top_k]


def rag_query(user_question, knowledge_base, embeddings_cache):

    return response.choices[0].message.content


def main():
    """
    Run example RAG queries with semantic search
    """
    print("=" * 70)
    print("RAG Assistant - Semantic Search with Embeddings")
    print("=" * 70)


if __name__ == "__main__":
    main()


