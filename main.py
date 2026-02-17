import os
import requests
from fastapi import FastAPI
# ... other imports ...

app = FastAPI()

# This pulls the token from Render's "Environment Variables" safely
HF_TOKEN = os.getenv("HF_TOKEN") 
API_URL = "https://router.huggingface.co/hf-inference/models/eli911/yola-business-ai"


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
