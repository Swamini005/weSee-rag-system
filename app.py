"""FastAPI service exposing POST /ask."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

import rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.ensure_index()  # builds the FAISS index on first run
    yield


app = FastAPI(title="WeSee Grounded Answer Engine", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class Citation(BaseModel):
    doc: str
    quote: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    answered: bool


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return AskResponse(**rag.answer_question(req.question))


@app.get("/health")
def health():
    return {"status": "ok"}
