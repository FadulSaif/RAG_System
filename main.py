import os
import sys
import arabic_reshaper
from bidi.algorithm import get_display
from PDF_Extraction import extract_and_chunk_pdf 
from vector_storage import save_to_vector_store, search_vector_store
from api_client import get_arabert_embedding, generate_jais_response

def print_arabic(text):
    """Formats Arabic text to display correctly in Windows terminals"""
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    print(bidi_text)

def main():
    print("--- Seela RAG System Initializing ---")
    
    # 1. Setup paths for the Dataset folder
    pdf_files = [
        os.path.join("Dataset", "legal-guide-to-childs-rights-in-libya-arabic.pdf"),
        os.path.join("Dataset", "libyan_law.pdf")
    ]
    
    # 2. Extract and Chunk Text
    print("Step 1: Extracting text from Libyan Legal PDFs...")
    all_chunks = []
    for pdf in pdf_files:
        if os.path.exists(pdf):
            print(f"   -> Processing: {pdf}")
            # ADDED: Smaller chunk sizes to prevent the AraBERT 512-token limit crash
            chunks = extract_and_chunk_pdf(pdf, chunk_size=250, overlap=50)
            all_chunks.extend(chunks)
        else:
            print(f"Warning: {pdf} not found. Please check the 'Dataset' folder.")

    if not all_chunks:
        print("Error: No text was extracted. Check your PDF paths and try again.")
        return

    # 3. Generate Embeddings and Save to FAISS
    print(f"Step 2: Generating AraBERT embeddings for {len(all_chunks)} chunks...")
    print("   (Note: This may take up to a minute if the Hugging Face API is in 'Cold Start' mode)")
    
    all_embeddings = []
    for i, chunk in enumerate(all_chunks):
        embedding = get_arabert_embedding(chunk)
        all_embeddings.append(embedding)
        if (i + 1) % 5 == 0:
            print(f"   -> Progress: {i + 1}/{len(all_chunks)} chunks embedded...")

    print("Step 3: Indexing vectors in local FAISS store...")
    save_to_vector_store(all_embeddings, all_chunks)
    print("System Ready!!!")

    # 4. Run the Required Test Questions
    test_questions = [
        "ما هي المسؤوليات الرئيسية للوزارات والجهات الحكومية في ليبيا فيما يخص رعاية الأطفال؟",
        "ما هي الأحكام القانونية المتعلقة بالوصاية والحضانة والمسؤوليات الأبوية وفقا للدليل؟",
        "ما هي المعاهدات الدولية المتعلقة بحقوق الطفل التي صادقت عليها ليبيا؟",
        "كم قيمة رأس مال المصرف؟",
        "ما هي المؤهلات التعليمية المشروطة للمدير العام ومجلس الإدارة؟"
     ]

    print("\n" + "="*50)
    print("--- RUNNING EVALUATOR TEST QUESTIONS ---")
    print("="*50)

    for question in test_questions:
        print_arabic(f"\n[USER QUESTION]: {question}")
        
        q_vec = get_arabert_embedding(question)
        retrieved_chunks = search_vector_store(q_vec, top_k=3)
        context = "\n\n".join(retrieved_chunks)
        
        answer = generate_jais_response(context, question)
        
        print_arabic(f"[LLM ANSWER]: {answer}")

    # 5. Interactive Mode
    print("\n" + "="*50)
    print("--- 💬 INTERACTIVE MODE (Type 'exit' or 'Click ctrl + C' to quit) ---")
    while True:
        user_input = input("\nAsk your own legal question: ")
        if user_input.lower() in ['exit', 'quit', 'خروج']:
            print("Shutting down. Good luck!")
            break
            
        if not user_input.strip(): continue

        q_vec = get_arabert_embedding(user_input)
        retrieved_chunks = search_vector_store(q_vec, top_k=3)
        context = "\n\n".join(retrieved_chunks)
        answer = generate_jais_response(context, user_input)
        
        print_arabic(f"\n[Answer]: {answer}")

if __name__ == "__main__":
    main()