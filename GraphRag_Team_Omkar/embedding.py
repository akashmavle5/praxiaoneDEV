from sentence_transformers import SentenceTransformer

class Embedder:
    """Wrapper class for generating text embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def encode(self, text: str) -> list[float]:
        """Generate vector embedding for the given text."""
        return self.model.encode(text).tolist()
