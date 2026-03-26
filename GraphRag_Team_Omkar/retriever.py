from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from graph_rag import get_graph_context
from pdf_ingestion import search_pdf
import os   
class GraphRetriever:
    """Module responsible for Neo4j database retrieval operations."""
    
    def __init__(self, graph_builder, embedder=None):
        self.graph = graph_builder
        self.embedder = embedder

    def retrieve_similar_diagnoses(self, query: str, top_k: int = 5) -> list[dict]:
        """Perform a semantic vector search across Diagnosis nodes."""
        if not self.embedder:
            raise ValueError("Embedder not configured for semantic search.")
            
        query_embedding = self.embedder.encode(query)

        cypher = """
        CALL db.index.vector.queryNodes('node_embedding_index', $top_k, $embedding)
        YIELD node, score
        RETURN node.name AS diagnosis, node.icd10 AS icd10, score
        """

        with self.graph.driver.session() as session:
            result = session.run(cypher, top_k=top_k, embedding=query_embedding)
            return [dict(r) for r in result]


# -------------------------------
# Embedding model (LIGHTWEIGHT)
# -------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------------
# Load vector database
# -------------------------------


if os.path.exists("vector_store"):
    db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
else:
    db = FAISS.from_texts(["empty"], embeddings)

# -------------------------------
# Medical LLM model
# -------------------------------

llm = Ollama(
    model="hf.co/RichardErkhov/m42-health_-_Llama3-Med42-8B-gguf:Q4_K_M",
    num_gpu=0
)



# -------------------------------
# LLM Query Function
# -------------------------------
chat_memory = []
def ask_llm(query):

    # -------- Graph Context --------
    graph_context = get_graph_context(query)

    # -------- Vector Search --------
    docs = db.similarity_search(query, k=4)

    file_context = "\n\n".join([d.page_content for d in docs])

    # Chat memory
    memory = "\n".join(chat_memory[-4:])

    context = f"""
CHAT HISTORY:
{memory}

GRAPH KNOWLEDGE:
{graph_context}

DOCUMENT KNOWLEDGE:
{file_context}
"""

    prompt = f"""
You are a clinical AI assistant.

Use ONLY the context provided.

If the answer is not found say:
"I could not find this information in the provided data."

Context:
{context}

Question:
{query}

Provide a clear medical explanation.
"""

    answer = llm.invoke(prompt)

    chat_memory.append(query)
    chat_memory.append(answer)

    return answer