from fastapi import FastAPI
from pydantic import BaseModel
from QueryProcessor import process_user_query

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = process_user_query(req.message)
    return ChatResponse(answer=answer)