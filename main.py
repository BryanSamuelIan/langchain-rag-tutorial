from fastapi import FastAPI
from pydantic import BaseModel
from query_data import query

app = FastAPI()

class QueryInput(BaseModel):
    query_text: str


@app.get("/")
def root():
    return {"message": "YUCCA v1 API is running"}


@app.post("/ask")
def ask(payload: QueryInput):
    result = query(payload.query_text)
    return result