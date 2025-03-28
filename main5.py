# main.py
import json
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional

from druid_service import suggest_druid_rollups_and_ingestion_spec

# Load API key from .env file
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Initialize the Hugging Face inference client
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", token=HF_API_KEY)
#client = InferenceClient(model="meta-llama/Llama-2-7b-chat-hf", token=HF_API_KEY)


app = FastAPI()

def get_ai_response(prompt: str):
    """Sends a request to the AI model and returns JSON if possible."""
    try:
        response = client.text_generation(prompt, max_new_tokens=500)
        return extract_json(response)
    except Exception as e:
        return {"error": str(e)}

def extract_json(text: str):
    """Extract JSON content from AI response."""
    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_text = text[json_start:json_end]
            return json.loads(json_text)
    except json.JSONDecodeError:
        pass
    return {}

def identify_fields(event_or_schema: Dict):
    """Identify key fields based on the event JSON schema."""
    prompt = f"""
    Analyze the following JSON event and identify key fields **only if they exist**.

    **Identification Rules:** 
    - **Timestamp Fields:** Identify fields that contain date-time values.
    - **De-duplication Key:** Identify a unique identifier.
    - **PII Fields:** Identify PII fields like email, phone, or name.

    **Strict Constraints:** Omit fields if not applicable.

    **Return JSON Format:** 
    {{
        "timestamp_fields": ["field1", "field2"],
        "de_duplication_key": "field_name",
        "pii_fields": ["field1", "field2"]
    }}

    **Event Data:**
    {json.dumps(event_or_schema, indent=2)}
    """
    return get_ai_response(prompt)

@app.post("/api/analyze-event/")
async def analyze_event(request: Request):
    """API endpoint to analyze event schema and return key field suggestions."""
    request_data = await request.json()
    if not isinstance(request_data, dict):
        return JSONResponse(content={"error": "Invalid JSON input"}, status_code=400)

    analysis_result = identify_fields(request_data)
    return JSONResponse(content=analysis_result)

@app.post("/api/suggest-dataset-name/")
async def suggest_dataset_name(request: Request):
    """API endpoint to suggest user-friendly dataset names based on schema and data analysis."""
    event_or_schema = await request.json()
    prompt = f"""
    You are an expert in data modeling and domain analysis. Based on the given event schema, analyze the data to identify its domain (e.g., user activity, IoT telemetry, financial transactions, product catalog, etc.). Generate 5 unique and user-friendly dataset names that reflect the data's purpose and domain.

    **Rules:**
    - Do not concatenate all field names into long names. Instead, focus on understanding the schema and the broader context.
    - Suggest names that are concise, descriptive, and easy to remember.
    - Ensure the names are relevant to common data categories such as logs, events, metrics, profiles, transactions, etc.
    - Analyze the structure and key properties dynamically to understand what the dataset might represent (e.g., user interactions, system metrics, content usage).
    - Identify key patterns, themes, and relationships in the data **without relying on specific field names**, as the event structure may vary.
    - Focus on clarity, simplicity, and real-world relevance based on the data’s potential purpose.
    - Avoid mechanically concatenating all keys or unrelated terms. Instead, aim for thoughtful, human-like naming that reflec

    **Return JSON in this format:**
    {{
        "dataset_names": ["name1", "name2", "name3", "name4", "name5"]
    }}

    **Event Schema:**
    {json.dumps(event_or_schema, indent=2)}
    """
    response = get_ai_response(prompt)
    return JSONResponse(content=response or {"dataset_names": ["Default_Dataset"]})

@app.post("/api/suggest-druid-rollups/")
async def suggest_druid_rollups(request: Request):
    """API endpoint to suggest Druid rollups and generate ingestion spec using AI."""
    event_or_schema = await request.json()

    # Call the AI-based Druid rollup suggestion service
    result = suggest_druid_rollups_and_ingestion_spec(event_or_schema)
    return JSONResponse(content=result)





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
