import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize the app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class BusinessData(BaseModel):
    data_input: str

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_PATH = "eli777/yola-business-ai" 
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_PATH}"

# 1. FIXING THE 404 ON THE HOME PAGE
@app.get("/")
def home():
    return {"message": "Yola AI Backend is Online!", "docs": "/docs"}

# 2. THE ANALYZE ROUTE
@app.post("/analyze")
async def analyze(data: BusinessData):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    smart_prompt = f"Analyze this business data: {data.data_input}"
    
    payload = {"inputs": smart_prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    return {"analysis": response.json()}
