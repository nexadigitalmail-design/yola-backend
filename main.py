import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# This allows your index.html file to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace with your actual details
HF_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/eli911/yola-business-ai"

class BusinessData(BaseModel):
    data_input: str

def get_cloud_advice(text):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": f"### Instruction:\nAnalyze business.\n\n### Input:\n{text}\n\n### Response:",
        "parameters": {"max_new_tokens": 300}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        # Extracting the text from the Hugging Face list response
        return response.json()[0]['generated_text'].split("### Response:")[-1]
    else:
        return f"Cloud Error: {response.status_code} - {response.text}"

@app.post("/analyze")
async def analyze(data: BusinessData):
    # This actually runs the AI function!
    advice = get_cloud_advice(data.data_input)
    return {"analysis": advice}