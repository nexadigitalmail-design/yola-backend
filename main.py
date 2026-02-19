import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Yola AI Business Engine")

# 1. ALLOW THE HANDSHAKE (CORS)
# This lets Lovable connect to your Koyeb URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BusinessData(BaseModel):
    data_input: str

HF_TOKEN = os.getenv("HF_TOKEN")
# TODO: Update with your specific model path
API_URL = "https://api-inference.huggingface.co/models/YOUR_USERNAME/yola-business-ai"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.get("/")
def health_check():
    return {"status": "Yola AI is alive and ready for entrepreneurs"}

@app.post("/analyze")
async def analyze(data: BusinessData):
    # SYSTEM PROMPT: This tells the Brain HOW to think.
    # It forces categorization, forecasting, and sentiment analysis.
    prompt_with_instructions = (
        "Instructions: Act as a Business Intelligence Expert. "
        "1. Categorize all expenses mentioned. "
        "2. Forecast next week's sales based on the trends provided. "
        "3. Identify inventory risks. "
        "4. Score customer sentiment (1-10).\n"
        f"Business Data: {data.data_input}"
    )

    payload = {
        "inputs": prompt_with_instructions,
        "parameters": {"max_new_tokens": 500, "temperature": 0.7}
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        if response.status_code == 503:
            return {"error": "AI is warming up. Please retry in 20 seconds."}
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    return {"analysis": response.json()}

