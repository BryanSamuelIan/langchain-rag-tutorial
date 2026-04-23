from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import os
import json
from datetime import datetime
import hashlib
import getpass

# Load .env
load_dotenv()

# Constants
CHROMA_PATH = "chroma"
MEMORY_DIR = "user_memories"
PROMPT_TEMPLATE = """
You are YUCCA, an AI assistant. Answer the question based on the following context and what you remember about the user:

Context from documents:
{context}

What you remember about the user:
{user_memory}

Recent conversation:
{chat_history}

---

Question: {question}

Answer naturally using the provided context. If you are not confident with the knowledge base given, you should say you cant answer/dont have the answer. Only use personal information if directly relevant to the question.
"""

# FastAPI init
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic input schema
class QueryInput(BaseModel):
    query_text: str
    user_id: Optional[str] = None
    debug: Optional[bool] = False


# Reuse your class
class MultiUserMemoryManager:
    def __init__(self, user_id=None):
        if not os.path.exists(MEMORY_DIR):
            os.makedirs(MEMORY_DIR)
        self.user_id = user_id or self._get_user_id()
        self.memory_path = os.path.join(MEMORY_DIR, f"{self.user_id}.json")
        self.user_memory = self.load_user_memory()
        self.chat_history = self.load_chat_history()

    def _get_user_id(self):
        import platform
        username = getpass.getuser()
        machine_name = platform.node()
        unique_string = f"{username}_{machine_name}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]

    def load_user_memory(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    return json.load(f).get('user_memory', {})
            except:
                return {}
        return {}

    def load_chat_history(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    return json.load(f).get('chat_history', [])
            except:
                return []
        return []

    def save_memory(self):
        with open(self.memory_path, 'w', encoding='utf-8') as f:
            json.dump({
                'user_id': self.user_id,
                'user_memory': self.user_memory,
                'chat_history': self.chat_history[-10:],
                'last_updated': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

    def extract_memory_from_message(self, message: str):
        extracted = {}
        message_lower = message.lower()
        name_patterns = [("panggil aku ", "name"), ("nama saya ", "name"), ("call me ", "name"), ("my name is ", "name"), ("i am ", "name")]
        for pattern, key in name_patterns:
            if pattern in message_lower:
                parts = message_lower.split(pattern)
                if len(parts) > 1:
                    name = parts[1].strip().split()[0]
                    if name:
                        extracted[key] = name
                        break
        if "umur saya" in message_lower or "saya berumur" in message_lower:
            for word in message_lower.split():
                if word.isdigit():
                    extracted["age"] = word
                    break
        if "saya tinggal di" in message_lower or "saya dari" in message_lower:
            parts = message_lower.split("saya tinggal di") if "saya tinggal di" in message_lower else message_lower.split("saya dari")
            if len(parts) > 1:
                location = parts[1].strip().split()[0]
                if location:
                    extracted["location"] = location
        return extracted

    def update_user_memory(self, message: str) -> Optional[str]:
        extracted = self.extract_memory_from_message(message)
        if extracted:
            self.user_memory.update(extracted)
            self.save_memory()
            ack_parts = []
            if 'name' in extracted:
                ack_parts.append(f"Baik, saya akan memanggil Anda {extracted['name']}")
            if 'age' in extracted:
                ack_parts.append(f"Saya akan mengingat bahwa umur Anda {extracted['age']} tahun")
            if 'location' in extracted:
                ack_parts.append(f"Saya akan mengingat bahwa Anda tinggal di {extracted['location']}")
            return ". ".join(ack_parts) + "." if ack_parts else None
        return None

    def add_to_chat_history(self, human_message, ai_message):
        self.chat_history.append({'human': human_message, 'ai': ai_message, 'timestamp': datetime.now().isoformat()})
        self.save_memory()

    def get_user_memory_text(self):
        return "\n".join([f"- {k}: {v}" for k, v in self.user_memory.items()]) if self.user_memory else "No personal information remembered."

    def get_chat_history_text(self):
        if not self.chat_history:
            return "No recent conversation history."
        history = []
        for item in self.chat_history[-3:]:
            history.append(f"Human: {item['human']}")
            history.append(f"AI: {item['ai']}")
        return "\n".join(history)

    def handle_personal_query(self, query: str):
        query_lower = query.lower()
        if any(p in query_lower for p in ["nama saya", "namaku", "siapa nama", "what is my name"]):
            return f"Nama Anda adalah {self.user_memory['name']}." if 'name' in self.user_memory else "Saya belum mengetahui nama Anda."
        if any(p in query_lower for p in ["umur saya", "berapa umur", "my age"]):
            return f"Umur Anda adalah {self.user_memory['age']} tahun." if 'age' in self.user_memory else "Saya belum mengetahui umur Anda."
        if any(p in query_lower for p in ["tinggal di mana", "saya tinggal di mana"]):
            return f"Anda tinggal di {self.user_memory['location']}." if 'location' in self.user_memory else "Saya belum mengetahui di mana Anda tinggal."
        return None


@app.post("/ask")
async def ask_query(payload: QueryInput):
    query_text = payload.query_text
    memory_manager = MultiUserMemoryManager(user_id=payload.user_id)

    if payload.debug:
        return {
            "debug": {
                "user_memory": memory_manager.user_memory,
                "chat_history": memory_manager.chat_history
            }
        }

    memory_response = memory_manager.update_user_memory(query_text)
    if memory_response:
        memory_manager.add_to_chat_history(query_text, memory_response)
        return {"response": memory_response, "debug": {"type": "memory_update"}}

    personal_response = memory_manager.handle_personal_query(query_text)
    if personal_response:
        memory_manager.add_to_chat_history(query_text, personal_response)
        return {"response": personal_response, "debug": {"type": "personal_query"}}

    # Vector search
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Format chunks
    similarity_chunks = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        }
        for doc, score in results
    ] if results else []

    user_memory_text = memory_manager.get_user_memory_text()
    chat_history_text = memory_manager.get_chat_history_text()

    # Decision
    if not results or results[0][1] < 0.7:
        greetings = ["hallo", "halo", "hi", "hello", "hey"]
        if query_text.lower() in greetings:
            name = memory_manager.user_memory.get("name")
            response_text = f"Halo {name}! YUCCA di sini. Apa yang bisa YUCCA bantu?" if name else "Halo! YUCCA di sini. Apa yang bisa YUCCA bantu?"
        else:
            response_text = "Maaf, YUCCA tidak tahu jawabannya."
        response_type = "fallback_response"
        prompt_used = None
    else:
        context_text = "\n\n---\n\n".join([chunk["content"] for chunk in similarity_chunks])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(
            context=context_text,
            question=query_text,
            user_memory=user_memory_text,
            chat_history=chat_history_text
        )
        model = ChatOpenAI()
        response_text = model.predict(prompt)
        response_type = "llm_response"
        prompt_used = prompt

    memory_manager.add_to_chat_history(query_text, response_text)

    return {
        "response": response_text,
        "sources": [chunk.get("metadata", {}).get("source") for chunk in similarity_chunks],
        "chunks_used": similarity_chunks,
        "debug": {
            "query_text": query_text,
            "user_memory_text": user_memory_text,
            "chat_history_text": chat_history_text,
            "type": response_type,
            "final_prompt": prompt_used
        }
    }


