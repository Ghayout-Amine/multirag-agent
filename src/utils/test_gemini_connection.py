import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}"
payload = {
    "contents": [
        {
            "parts": [
                {"text": "Reponds en un seul mot: Pret"}
            ]
        }
    ]
}

resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
print("STATUS:", resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    answer = data["candidates"][0]["content"]["parts"][0]["text"]
    print("REPONSE GEMINI 3.5 FLASH LITE:", answer.strip())
else:
    print("ERROR:", resp.text)
