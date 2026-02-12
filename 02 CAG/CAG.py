"""
Project 2: CAG - Cache Augmented Generation (OpenAI Prompt Caching)

"""

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = InferenceClient(token=os.getenv("HUGGING_FACE_API_KEY"))

chat_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HUGGING_FACE_API_KEY")
)

def load_knowledge_base(knowledge_dir: str = "knowledge") -> str:
    """
    STEP 1: Load all documents and combine them into ONE big string
    Unlike RAG (which keeps docs separate), CAG combines everything!
    """
    print("\n" + "="*70)
    print("STEP 1: Loading Knowledge Base")
    print("="*70)

    knowledge_parts = []
    knowledge_path = Path(__file__).parent / knowledge_dir

    if not knowledge_path.exists():
        print(f"Error: knowledge directory not found: {knowledge_path}")
        return ""
    
    for file_path in sorted(knowledge_path).glob("*.md"):
        doc_id = file_path.stem
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            knowledge_parts.append(f"=== {doc_id.upper().replace('_', "")} ===\n{content}")
            print(f"Loaded {doc_id}")

    full_context = "\n\n".join(knowledge_parts)
    print(f"\nTotal: {len(full_context)} characters combined")

    return full_context

def cag_query(user_question: str, knowledge_context: str) -> str:
    """
    STEP 2: Send question with ALL knowledge in the system prompt
    The system prompt gets cached automatically by OpenAI!
    """
    
    return response.choices[0].message.content


def main():
    """
    Simple CAG Demo - Load once, query many times!
    """
    print("="*70)
    print("CloudTech CAG Assistant - Simple Version")
    print("="*70)
    

if __name__ == "__main__":
    main()
