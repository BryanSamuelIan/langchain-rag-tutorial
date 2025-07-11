from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# FastAPI app
app = FastAPI()

# Pydantic model for request body
class Query(BaseModel):
    query_text: str

# Load embeddings & DB once at startup
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

@app.post("/ask")
async def ask_question(query: Query):
    query_text = query.query_text

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Fallback if not enough relevant info
    if len(results) == 0 or results[0][1] < 0.7:
        greetings = ["hallo", "halo", "hi", "hello", "hey"]
        if query_text.lower() in greetings:
            return {
                "response": "Halo! YUCCA di sini. Apa yang bisa YUCCA bantu?",
                "sources": []
            }
        return {
            "response": "Sorry, YUCCA tidak mengetahui jawaban untuk pertanyaan itu.",
            "sources": []
        }

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI()
    response_text = model.predict(prompt)
    sources = [doc.metadata.get("source", None) for doc, _ in results]

    return {
        "response": response_text,
        "sources": sources
    }
