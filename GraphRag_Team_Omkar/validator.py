import json
import re
from pydantic import BaseModel, ValidationError
from typing import List, Optional

class MedicalEntity(BaseModel):
    category: str
    entity_value: str

class EncounterEntities(BaseModel):
    entities: Optional[List[MedicalEntity]] = None
    
def validate_llm_json(raw_output):
    """Safely cleans and validates the JSON coming from the LLM, ignoring conversational filler."""
    try:
        data = raw_output
        if isinstance(raw_output, str):
            # Find everything from the first '{' to the last '}'
            match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            
            if match:
                clean_text = match.group(0) # Extract just the JSON block
                data = json.loads(clean_text)
            else:
                raise ValueError("No JSON object could be found in the LLM response.")
                
        # Validate through Pydantic (allow dynamic extraction parsing)
        validated = EncounterEntities(**data)
        return validated.dict() 
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output into JSON. Error: {str(e)}\nRaw Output: {raw_output}")
    except ValidationError as e:
        raise ValueError(f"Schema validation error on LLM output: {str(e)}")