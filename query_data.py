import argparse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Function to search the Chroma database
def search_db(query_text, db):
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return None
    return results

# Function to generate the prompt using the context
def generate_prompt(results, query_text):
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    return prompt

# Function to get a response from Hugging Face's text generation model
def get_response(prompt):
    model = pipeline("text-generation", model="bigscience/bloom-560m")  # Replace with your Hugging Face model
    response = model(prompt, max_length=100, num_return_sequences=1)
    return response[0]["generated_text"]

# Function to load the Chroma DB (load only once)
def load_chroma_db():
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Hugging Face model for embeddings
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return db

# Main function to handle the flow
def main():
    # Create CLI for query input
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load the Chroma DB (do this once)
    db = load_chroma_db()

    # Search the DB
    results = search_db(query_text, db)
    if results is None:
        return

    # Generate the prompt based on search results
    prompt = generate_prompt(results, query_text)

    # Get the response from the model
    response_text = get_response(prompt)

    # Collect the sources from the results
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
