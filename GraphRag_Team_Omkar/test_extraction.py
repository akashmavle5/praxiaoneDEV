import traceback
import sys
sys.path.append(r'e:\updated_internship_gemini')
from pdf_ingestion import extract_text

try:
    print("Extracting...")
    text = extract_text(r'e:\updated_internship_gemini\Individual Medicalim.pdf')
    print('Text length:', len(text))
    if text:
        print('Content[:300]:', text[:300].encode('utf-8'))
    else:
        print('No text')
except Exception as e:
    print('Error:')
    traceback.print_exc()
