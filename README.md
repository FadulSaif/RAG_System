# Seela AI - RAG System

1. Instructions on How to Run the Code

1. Ensure you have Python installed (Fully compatible with Python 3.8 through 3.13).

2. Install the required dependencies by running:
pip install -r requirements.txt

3. Create a .env file in the root directory and paste your Hugging Face Access Token:
HUGGINGFACE_API_KEY=your_token_here

4. Ensure the dataset PDFs (legal-guide-to-childs-rights-in-libya-arabic.pdf and libyan_law.pdf) are located exactly inside a folder named Dataset.

5. Run the application:
python main.py


2. Approach and Design Decisions
1. Modularity: The codebase is strictly modularized (PDF_Extraction.py, vector_storage.py, api_client.py) to separate data ingestion, vector storage, and API calling logic from the main execution thread.

2. Official Hub Integration & Fault Tolerance: To ensure maximum stability and bypass raw HTTP routing limits, the text generation utilizes the official huggingface_hub InferenceClient. It features an automated try-except fallback architecture: it prioritizes the required Jais-adapted-7b model, but if the Hugging Face server is overwhelmed or the model is offline, it automatically reroutes the prompt to Qwen-2.5-7B (a highly available, Arabic-fluent model) to guarantee uninterrupted execution.

3. Local Vector Database (FAISS): To strictly meet the requirement and to ensure full compatibility, I utilized FAISS instead of Milvus Lite (Which was planned but not anymore due to compatibility with python 3.13). It runs entirely in-memory and saves to a local .index file, requiring zero Docker or server configuration.

4. Token Limit Management: AraBERT possesses a strict 512-token limit. The PDF extraction logic calculates optimal chunk sizes (250 words with a 50-word overlap) to ensure deep contextual retrieval without crashing the embedding model's tensor limits.

5. Arabic Text Handling: Because standard PDF extraction and Windows terminals scramble Right-to-Left languages, I utilized arabic_reshaper and python-bidi to ensure the Arabic text is perfectly formatted both for the LLM's context window and the terminal output.

6. Strict Prompt Engineering: The system prompt explicitly forces the LLM to only answer based on the retrieved chunks to prevent hallucination, fulfilling the requirement to generate responses from the provided documents.


3. Limitations
1. Hugging Face Free Tier: Per the requirements, this project strictly uses free LLM APIs via Hugging Face Serverless Inference. You may experience "Cold Start" delays (taking 30-60 seconds) on the very first query while the server wakes the model up.

2. Rate Limiting & Server Availability: Free APIs are subject to strict rate limiting and global traffic loads. While the automated fallback logic mitigates downtime for the generation model, if the AraBERT embedding endpoint fails due to traffic, the script may need to be paused and rerun.

3. Model Constraints: Because we are restricted to Serverless Inference, we cannot easily adjust lower-level model parameters (like repetition penalties) as freely as we could on a dedicated, paid inference endpoint.