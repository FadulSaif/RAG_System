import requests
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Headers for the AraBERT request
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

# The working AraBERT URL
EMBEDDING_API_URL = "https://router.huggingface.co/hf-inference/models/aubmindlab/bert-base-arabertv02/pipeline/feature-extraction"

# Initialize the official Hugging Face Client for text generation
hf_client = InferenceClient(api_key=HF_API_KEY)

def get_arabert_embedding(text):
    payload = {"inputs": text}
    try:
        response = requests.post(EMBEDDING_API_URL, headers=HEADERS, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list): 
                    return result[0]
                return result
            return result
        else:
            print(f"Embedding Error {response.status_code}: {response.text}")
            return [0.0] * 768
    except Exception as e:
        print(f"Request failed: {e}")
        return [0.0] * 768

def generate_jais_response(context, question):
    prompt = f"أنت مساعد قانوني ذكي. استخدم السياق التالي للإجابة.\nالسياق: {context}\nالسؤال: {question}"
    
    try:
        # 1. Try Jais (Follows the strict technical test guidelines)
        completion = hf_client.chat.completions.create(
            model="inceptionai/jais-adapted-7b-chat", 
            messages=[{"role": "user", "content": prompt}], 
            max_tokens=300
        )
        return completion.choices[0].message.content
        
    except Exception as e:
        # 2. If Jais is offline, use the official auto-routed Qwen fallback
        print("\n   [Warning] Jais model is currently offline. Auto-routing to Qwen-7B...")
        try:
            # Qwen 2.5 7B is massively supported on HF's free tier and fluent in Arabic
            completion = hf_client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct", 
                messages=[{"role": "user", "content": prompt}], 
                max_tokens=300
            )
            return completion.choices[0].message.content
        except Exception as e2:
            return f"Fallback LLM Error: {e2}"