from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import pytesseract
from PIL import Image
import docx
import pdfplumber
from pdf2image import convert_from_path
import os

def ingest_pdf(path):
    try:
        # Load PDF
        loader = PyPDFLoader(path)
        pages = loader.load()

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(pages)

        # Load embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create vector store
        db = FAISS.from_documents(chunks, embeddings)

        # Save vector database
        db.save_local("vector_store")

        print("PDF Indexed Successfully")

        return "PDF ingested successfully"

    except Exception as e:
        print("PDF ingestion error:", e)
        return "PDF ingestion failed"


def extract_text(file_path):

    if file_path.endswith(".pdf"):

        text = ""

        # Try normal PDF extraction
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
        except:
            pass

        # If no text found → scanned PDF → OCR
        if text.strip() == "":
            images = convert_from_path(file_path)

            for img in images:
                text += pytesseract.image_to_string(img)

        return text


    elif file_path.endswith(".txt"):

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


    elif file_path.endswith(".docx"):

        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])


    elif file_path.endswith((".png", ".jpg", ".jpeg")):

        return pytesseract.image_to_string(Image.open(file_path))


    else:
        return ""

def search_pdf(question, k=3):

    try:

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if not os.path.exists("vector_store"):
            return "No documents indexed yet."

        db = FAISS.load_local(
            "vector_store",
            embeddings,
            allow_dangerous_deserialization=True
        )

        docs = db.similarity_search(question, k=k)

        return "\n\n".join([d.page_content for d in docs])

    except Exception as e:
        return f"Document search error: {str(e)}"