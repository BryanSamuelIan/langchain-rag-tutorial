from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil
import json
from typing import List

# To access administrator mode
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/UC"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
    save_chunks_to_file(chunks)

# Load .md files to documents
def load_documents() -> List[Document]:
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

# Split text into chunks with respect to token limit
def split_text(documents: List[Document], chunk_size: int = 1000, overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Print details about the 10th chunk if available
    document = chunks[10] if len(chunks) > 10 else chunks[0]
    print(f"Example Chunk Content: {document.page_content}")
    print(f"Example Chunk Metadata: {document.metadata}")

    return chunks

# Save chunks to Chroma DB
def save_to_chroma(chunks: List[Document]):
    # Clear out the database first (if it exists).
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Initialize Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# Save chunks to a JSON file
def save_chunks_to_file(chunks: List[Document], filename="chunks.json"):
    # Convert chunks to a list of dictionaries for saving as JSON
    chunks_data = [{"content": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks]

    # Save to a JSON file
    with open(filename, "w") as f:
        json.dump(chunks_data, f, indent=4)
    print(f"Chunks saved to {filename}")

if __name__ == "__main__":
    main()
