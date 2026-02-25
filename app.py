import streamlit as st
import os
from PDF_Extraction import extract_and_chunk_pdf 
from vector_storage import save_to_vector_store, search_vector_store
from api_client import get_arabert_embedding, generate_jais_response

# --- UI Configuration ---
st.set_page_config(page_title="Seela AI Legal RAG", page_icon="⚖️", layout="centered")

# --- Custom HTML for Native Web Arabic ---
def rtl_text(text):
    # Web browsers handle Arabic natively. We just tell it the direction is RTL.
    return f"<div dir='rtl' style='text-align: right; font-size: 18px; font-family: Arial, sans-serif;'>{text}</div>"

st.markdown(rtl_text("<h1>⚖️ مساعد الذكاء الاصطناعي القانوني (Seela AI)</h1>"), unsafe_allow_html=True)
st.markdown(rtl_text("اسأل أي سؤال حول حقوق الطفل والقوانين الليبية."), unsafe_allow_html=True)
st.divider()

# --- Initialization Phase ---
@st.cache_resource
def initialize_system():
    # If the database already exists, skip building it
    if os.path.exists("vector_store.index"):
        return True 

    pdf_files = [
        os.path.join("Dataset", "legal-guide-to-childs-rights-in-libya-arabic.pdf"),
        os.path.join("Dataset", "libyan_law.pdf")
    ]
    
    all_chunks = []
    for pdf in pdf_files:
        if os.path.exists(pdf):
            chunks = extract_and_chunk_pdf(pdf, chunk_size=200, overlap=50)
            all_chunks.extend(chunks)
            
    if not all_chunks:
        return False

    all_embeddings = []
    for chunk in all_chunks:
        all_embeddings.append(get_arabert_embedding(chunk))

    save_to_vector_store(all_embeddings, all_chunks)
    return True

with st.spinner("جاري تهيئة قاعدة البيانات... (Building fresh database...)"):
    success = initialize_system()
    if not success:
        st.error("❌ Error: Could not find PDFs in the Dataset folder.")
        st.stop()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(rtl_text(message["content"]), unsafe_allow_html=True)

if prompt := st.chat_input("اكتب سؤالك القانوني هنا... (Type your legal question...)"):
    
    with st.chat_message("user"):
        st.markdown(rtl_text(prompt), unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث وتوليد الإجابة... (Searching & Generating...)"):
            # Execute RAG Pipeline
            q_vec = get_arabert_embedding(prompt)
            retrieved_chunks = search_vector_store(q_vec, top_k=3)
            context = "\n\n".join(retrieved_chunks)
            
            answer = generate_jais_response(context, prompt)
            
            # Print Final Answer
            st.markdown(rtl_text(answer), unsafe_allow_html=True)
            
            # Show Context Expandable Box
            with st.expander("🔍 عرض المصادر (View Retrieved Context)"):
                st.markdown(rtl_text(context), unsafe_allow_html=True)
                
    st.session_state.messages.append({"role": "assistant", "content": answer})