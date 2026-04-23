from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """
You are YUCCA, the official AI chatbot of Universitas Ciputra.

Your role:
- Help users with accurate information about Universitas Ciputra
- Answer in a friendly, natural, and helpful tone
- Use ONLY the provided context as your main knowledge source
- Answer in the language the user ask you

Rules:
- If the answer is NOT in the context, say you don't know
- DO NOT hallucinate or make up information
- Keep answers concise but informative
- If user greets, respond warmly as YUCCA
- If relevant, guide the user (e.g., suggest programs, next steps)
- Make numbers in natural language spelling (e.g one thousand), not numeral (e.g 1000)

Context:
{context}
"""),
    ("user", "Question: {question}")
])


def query(query_text: str):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Handle no results
    if not results or results[0][1] < 0.7:
        greetings = ["hallo", "halo", "hi", "hello", "hey"]
        if query_text.lower() in greetings:
            return {
                "response": "Halo! YUCCA di sini 👋 Apa yang bisa saya bantu terkait Universitas Ciputra?",
                "sources": []
            }

        return {
            "response": "Maaf, YUCCA belum memiliki informasi untuk menjawab pertanyaan tersebut.",
            "sources": []
        }

    # Build context
    context_text = "\n\n---\n\n".join([
        doc.page_content for doc, _ in results
    ])

    # Build prompt (structured, not string!)
    prompt = PROMPT_TEMPLATE.invoke({
        "context": context_text,
        "question": query_text
    })

    # LLM
    model = ChatOpenAI(model="gpt-4.1", temperature=0)
    response = model.invoke(prompt)

    response_text = response.content

    sources = [
        doc.metadata.get("source", None)
        for doc, _ in results
    ]

    return {
        "response": response_text,
        "sources": sources
    }