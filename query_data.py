import argparse
from dotenv import load_dotenv
load_dotenv()
# from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


# def query(query_text: str):
#     # Create CLI.
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text
#
#     # Prepare the DB.
#     embedding_function = OpenAIEmbeddings()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
#
#     # Search the DB.
#     results = db.similarity_search_with_relevance_scores(query_text, k=3)
#     if len(results) == 0 or results[0][1] < 0.7:
#         # If no results found, set response to a default message
#         response_text = "Sorry, YUCCA tidak mengetahui jawaban untuk pertanyaan itu."
#         # Handle greetings
#         greetings = ["hallo", "halo", "hi", "hello", "hey"]
#         if query_text.lower() in greetings:
#             response_text = "Halo! YUCCA di sini. Apa yang bisa YUCCA bantu?"
#     else:
#         context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#         prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#         prompt = prompt_template.format(context=context_text, question=query_text)
#         print(prompt)
#
#         model = ChatOpenAI(model="gpt-5", temperature=0)
#         response_text = model.invoke(prompt)
#         # response = model.invoke(prompt)
#         # response_text = response.content if hasattr(response, "content") else response
#
#     sources = [doc.metadata.get("source", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)
def query(query_text: str):
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0 or results[0][1] < 0.7:
        response_text = "Sorry, YUCCA tidak mengetahui jawaban untuk pertanyaan itu."
        greetings = ["hallo", "halo", "hi", "hello", "hey"]
        if query_text.lower() in greetings:
            response_text = "Halo! YUCCA di sini. Apa yang bisa YUCCA bantu?"
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=context_text,
            question=query_text
        )

        model = ChatOpenAI(model="gpt-5", temperature=0)
        response = model.invoke(prompt)
        response_text = response.content

    sources = [doc.metadata.get("source", None) for doc, _ in results]

    return {
        "response": response_text,
        "sources": sources
    }

if __name__ == "__main__":
    query()
