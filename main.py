import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add docs_url and redoc_url explicitly to fix the 404 Undocumented issue
app = FastAPI(
    title="Yola AI",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class BusinessData(BaseModel):
    data_input: str

# Use an environment variable for safety
HF_TOKEN = os.getenv("HF_TOKEN")
# Make sure this matches your HF username/model
MODEL_PATH = "eli777/yola-business-ai" 
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_PATH}"

@app.get("/")
def health():
    return {"status": "online", "message": "Visit /docs for API testing"}

@app.post("/analyze")
async def analyze(data: BusinessData):
    # Your logic here...
    return {"status": "Success", "data": data.data_input}

