def extract_medical_entities(text, query=None):
    try:
        from transformers import pipeline
        
        # Using a public Biomedical NER model
        # Alternatively, BioBERT fine-tuned on biomedical NER datasets can be used.
        # Model: d4data/biomedical-ner-all 
        ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
        
        # Combine query and context for extraction if present
        combined_text = text
        if query and query.strip():
             combined_text = f"Query: {query}\n\nContext: {text}"
             
        # Truncate text to avoid token limits for BERT
        truncated = combined_text[:2000] 
        results = ner_pipeline(truncated)
        
        if not results:
            return "**🔬 Clinical NER Extraction (PubMedBERT)**\n\nNo significant medical entities found in the document snippet."
        
        output = "**🔬 Clinical NER Extraction (PubMedBERT)**\n\n"
        output += "| Entity Group | Extracted Word | Confidence Score |\n|---|---|---|\n"
        
        for r in results:
            score_formatted = f"{(r['score'] * 100):.1f}%"
            output += f"| {r['entity_group']} | {r['word']} | {score_formatted} |\n"
            
        output += "\n\n*(Note: Extraction is based on the NLP4Science/pubmedbert-NER PubMedBERT model)*"
        return output
        
    except ImportError:
        return """**⚠️ Missing Dependencies for Clinical NER**

To use the PubMedBERT extraction pipeline, you need to install HuggingFace transformers.

Please run the following commands in your terminal:
```bash
pip install transformers torch
```
Once installed, try your query again!"""
    except Exception as e:
        return f"**NER Pipeline Error:** {e}"
