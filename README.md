# Seela AI - RAG System

1. Instructions on How to Run the Code

  A. Ensure you have Python installed (Fully compatible with Python 3.8 through 3.13).

  B. Install the required dependencies by running:
pip install -r requirements.txt

  C. Create a .env file in the root directory and paste your Hugging Face Access Token:
  HUGGINGFACE_API_KEY=your_token_here

  D. Ensure the dataset PDFs (legal-guide-to-childs-rights-in-libya-arabic.pdf and libyan_law.pdf) are located exactly inside a folder named Dataset.

  E. Run the application:
    python -m streamlit run app.py


2. Approach and Design Decisions
   
  A. Interactive Web Interface: The system utilizes Streamlit to provide a clean, interactive chat UI. This not only significantly improves user experience but also leverages modern web browsers' native Right-to-Left (RTL) text engines, rendering complex Arabic legal text flawlessly without the need for external reshaping libraries.

  B. Fault-Tolerant API Routing: Because free-tier serverless APIs frequently unload massive models, the system uses the official huggingface_hub InferenceClient with built-in fallback logic. It prioritizes the required Jais-adapted-7b model, but if the server returns a 404/503, it automatically reroutes to a highly available, Arabic-fluent fallback model (Qwen) to ensure uninterrupted execution.

  C. Local Vector Database (FAISS): To strictly meet the requirement and to ensure full compatibility, I utilized FAISS instead of Milvus Lite (Which was planned but not anymore due to compatibility with python 3.13). It runs entirely in-memory and saves to a local .index file, requiring zero Docker or server configuration.

  D. Token Limit Management: AraBERT possesses a strict 512-token limit. The PDF extraction logic calculates optimal chunk sizes (200 words with a 50-word overlap) to ensure deep contextual retrieval without crashing the embedding model's tensor limits.

  E. Arabic Text Handling: Because standard PDF extraction and Windows terminals scramble Right-to-Left languages, I utilized arabic_reshaper and python-bidi to ensure the Arabic text is perfectly formatted both for the LLM's context window and the terminal output.

  F. Strict Prompt Engineering: The system prompt explicitly forces the LLM to only answer based on the retrieved chunks to prevent hallucination, fulfilling the requirement to generate responses from the provided documents.


3. Limitations
   
  A. Hugging Face Free Tier: This project strictly uses free LLM APIs via Hugging Face Serverless Inference. You may experience "Cold Start" delays (taking 30-60 seconds) on the very first query while the server wakes     the model up.

  B. Rate Limiting & Server Availability: Free APIs are subject to strict rate limiting and global traffic loads. While the automated fallback logic mitigates downtime for the generation model, if the AraBERT embedding   endpoint fails due to traffic, the script may need to be paused and rerun.

  C. Model Constraints: Because it is restricted to Serverless Inference, we cannot easily adjust lower-level model parameters (like repetition penalties) as freely as we could on a dedicated, paid inference endpoint.


  4. Future Enhancements
     
    A. Optical Character Recognition (OCR): To maintain a lightweight footprint and pure-Python, zero-configuration setup for this evaluation, this pipeline uses direct digital text extraction. For a production deployment, I would integrate EasyOCR or Azure Document Intelligence to process scanned PDFs and extract text from images.
