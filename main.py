import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Yola AI Business Engine")

# 1. ENABLE CORS (Essential for Lovable connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BusinessData(BaseModel):
    data_input: str

# 2. Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
# IMPORTANT: Replace with your actual Hugging Face model path
MODEL_PATH = "eli777/yola-business-ai" 
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_PATH}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.get("/")
def health_check():
    return {"status": "Yola AI Engine is Online"}

@app.post("/analyze")
async def analyze(data: BusinessData):
    # This "Smart Prompt" tells the AI exactly what to do with the data
    smart_prompt = (
        "Instructions: You are a Business Analyst for SMEs in Nigeria. "
        "From the following data: 1. Categorize all expenses. 2. Forecast next week's sales. "
        "3. Identify inventory risks. 4. Analyze customer sentiment and product performance.\n\n"
        f"Business Data: {data.data_input}"
    )

    payload = {
        "inputs": smart_prompt,
        "parameters": {"max_new_tokens": 500, "temperature": 0.7}
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        if response.status_code == 503:
            return {"error": "AI is warming up on Hugging Face. Please retry in 20 seconds."}
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    return {"analysis": response.json()}


