import fitz  # PyMuPDF

def extract_and_chunk_pdf(file_path, chunk_size=250, overlap=50):
    """
    Extracts raw text from a PDF and splits it into manageable chunks.
    No text reshaping is done here to preserve the raw string for the browser.
    """
    try:
        # 1. Open the PDF
        doc = fitz.open(file_path)
        full_text = ""
        
        # 2. Extract raw text
        for page in doc:
            full_text += page.get_text() + "\n"
            
        # 3. Chunk the text
        words = full_text.split()
        chunks = []
        
        for i in range(0, len(words), max(1, chunk_size - overlap)):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []