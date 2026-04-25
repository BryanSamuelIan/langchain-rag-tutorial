from fastapi import FastAPI
from pydantic import BaseModel
from query_data import query
from contextlib import asynccontextmanager
import subprocess
import sys

class QueryInput(BaseModel):
    query_text: str

# app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up application...")
    
    try:
        print("📦 Running create_database.py...")

        result = subprocess.run(
            [sys.executable, "create_database.py"],  # safer than "python"
            capture_output=True,
            text=True
        )

        print("✅ Script finished")

        print("---- STDOUT ----")
        print(result.stdout)

        print("---- STDERR ----")
        print(result.stderr)

        print(f"Return code: {result.returncode}")

        if result.returncode != 0:
            raise RuntimeError("create_database.py failed")

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise e

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    
    return {"message": "YUCCA v1 API is running"}


@app.post("/ask")
def ask(payload: QueryInput):
    result = query(payload.query_text)
    return result