import json
from openai import OpenAI

# -------------------------------
# LLM Client (Ollama / Med42)
# -------------------------------

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

SYSTEM_PROMPT = """
You are a specialized medical reasoning engine building a Patient Journey Knowledge Graph (PJKG).

Extract structured medical information from the transcript.

Return STRICT JSON only.

Schema:

{
  "Diagnosis": [{"name": "string", "icd10": "string", "caused_by": "string"}],
  "Symptom": [{"name": "string", "severity": "string"}],
  "Medication": [{"name": "string", "dosage": "string", "intent": "string"}],
  "VitalSign": {
      "blood_pressure": "string",
      "heart_rate": "string",
      "weight": "string"
  },
  "Temporal": {
      "has_followup": true,
      "followup_timeline": "string"
  }
}

Return JSON only. No explanations.
"""

def extract_encounter_logic(transcript: str):

    print("\n" + "="*50)
    print("🧠 Sending transcript to Medical LLM...")
    print("="*50)

    response = client.chat.completions.create(
        model="hf.co/RichardErkhov/m42-health_-_Llama3-Med42-8B-gguf:Q4_K_M",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript}
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content

    print("📥 LLM Raw Output:")
    print(raw_output)

    # -------------------------------
    # Extract JSON safely
    # -------------------------------

    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1

        json_str = raw_output[start:end]

        return json.loads(json_str)

    except Exception as e:
        print("⚠ JSON parsing failed:", e)

        return {
            "Diagnosis": [],
            "Symptom": [],
            "Medication": [],
            "VitalSign": {},
            "Temporal": {}
        }