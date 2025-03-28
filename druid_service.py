import json
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from typing import Dict

# Initialize the Hugging Face inference client (Load from your existing client)
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Initialize the Hugging Face inference client
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", token=HF_API_KEY)


def suggest_druid_rollups_and_ingestion_spec(event_or_schema: Dict):
    """Use AI to suggest Druid rollups and generate an ingestion spec."""
    prompt = f"""
    Analyze the following JSON event schema and suggest appropriate rollups for Druid.
    Also, generate a valid Druid ingestion spec template.

    **Your tasks:**
    1. Suggest rollups for common aggregations (e.g., hourly, daily).
    2. Identify numeric fields for metrics aggregation.
    3. Identify non-numeric fields as dimensions.
    4. Detect timestamp fields and use them in the `timestampSpec`.

    **Event Schema:**
    {json.dumps(event_or_schema, indent=2)}

    **Expected JSON Response:**
    {{
        "rollup_suggestions": ["hourly_rollup", "daily_rollup", "custom_rollup"],
        "druid_ingestion_spec": {{
            "type": "index",
            "spec": {{
                "dataSchema": {{
                    "dataSource": "dynamic_data_source",
                    "timestampSpec": {{
                        "column": "timestamp_field",
                        "format": "auto"
                    }},
                    "dimensionsSpec": {{
                        "dimensions": ["dim1", "dim2"]
                    }},
                    "metricsSpec": [
                        {{"type": "doubleSum", "name": "metric1", "fieldName": "metric1"}}
                    ],
                    "granularitySpec": {{
                        "type": "uniform",
                        "segmentGranularity": "hour",
                        "queryGranularity": "none"
                    }}
                }},
                "ioConfig": {{
                    "type": "index",
                    "inputSource": {{
                        "type": "inline",
                        "data": [{json.dumps(event_or_schema)}]
                    }},
                    "inputFormat": {{
                        "type": "json"
                    }}
                }},
                "tuningConfig": {{
                    "type": "index",
                    "maxRowsInMemory": 100000,
                    "maxRowsPerSegment": 5000000
                }}
            }}
        }}
    }}
    """
    try:
        # Send the prompt to the AI model and extract JSON response
        response = client.text_generation(prompt, max_new_tokens=500)
        print(response)
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
