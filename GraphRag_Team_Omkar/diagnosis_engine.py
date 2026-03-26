from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

def diagnose(query):

    embeddings = HuggingFaceEmbeddings()

    db = FAISS.load_local("vector_store", embeddings)

    docs = db.similarity_search(query, k=3)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Based on the following discharge summary:

    {context}

    Provide possible diagnosis.
    """

    response = ollama.chat(
        model="hf.co/RichardErkhov/m42-health_-_Llama3-Med42-8B-gguf:Q4_K_M",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]