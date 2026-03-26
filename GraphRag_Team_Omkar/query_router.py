from langchain_community.llms import Ollama
import os

llm = Ollama(model="hf.co/RichardErkhov/m42-health_-_Llama3-Med42-8B-gguf:Q4_K_M")

UPLOAD_FOLDER = "uploads"


def get_available_files():
    if not os.path.exists(UPLOAD_FOLDER):
        return []
    return os.listdir(UPLOAD_FOLDER)


def classify_query(question):

    files = get_available_files()

    files_text = ", ".join(files) if files else "No files uploaded"

    prompt = f"""
You are an AI query router for a medical system.

Your job is to decide where the answer should come from.
You must output exactly ONE word from the list below: FILE, GRAPH, or AUTO.

DATA SOURCES:

1️⃣ FILE
Use this if the user mentions:
- "tabulate", "table", "extract"
- "document", "pdf", "file", "attached", "discharge summary"
- Any request to summarize or read uploaded files.

Uploaded files:
{files_text}

2️⃣ GRAPH
Use this if the user asks about:
- patient dataset, statistics, counts
- structured database queries or relationships

3️⃣ AUTO
Use this if the question is general medical knowledge not related to the uploaded files.

CRITICAL EXAMPLES:

User: what diet should patient follow from discharge report
Answer: FILE

User: Tabulate the discharge summaries attached to this document and also add diagnosis, treatment plan
Answer: FILE

User: Tabulate the files
Answer: FILE

User: i just uploaded the pdf, for it can you give me discharge medications
Answer: FILE

User: how many diabetic patients are there
Answer: GRAPH

User: what causes diabetes
Answer: AUTO

User Question:
{question}

Respond with ONLY one word: FILE, GRAPH, or AUTO.
"""

    response = llm.invoke(prompt)

    intent_text = response.strip().upper()

    if "FILE" in intent_text:
        return "FILE"
    elif "GRAPH" in intent_text:
        return "GRAPH"
    else:
        return "AUTO"