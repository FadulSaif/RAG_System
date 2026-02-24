import fitz
import arabic_reshaper
from bidi.algorithm import get_display
import os

def extract_and_chunk_pdf(pdf_path, chunk_size=250, overlap=50):
    if not os.path.isfile(pdf_path):
        print(f"File '{pdf_path}' does not exist.")
        return []
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return []
    
    full_text = []
    for page in doc:
        try:
            raw_text = page.get_text()
            reshaped_text = arabic_reshaper.reshape(raw_text)
            bidi_text = get_display(reshaped_text)
            full_text.append(bidi_text)
        except Exception as e:
            print(f"Error processing page: {e}")
    
    complete_text = " ".join(full_text)
    words = complete_text.split()
    chunks = []

    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))

    return chunks