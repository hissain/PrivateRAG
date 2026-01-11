import streamlit as st
import os
import tempfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- App Config ---
st.set_page_config(page_title="PrivateRAG (Cloud)", page_icon="🔒", layout="centered")

# --- Security: Password Protection ---
def check_password():
    """Returns `True` if the user had the correct password."""
    # Check if password is set in secrets
    if "APP_PASSWORD" not in st.secrets:
        st.error("⚠️ APP_PASSWORD not configured in Secrets!")
        return False

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Safe access to avoid KeyError if key was already deleted
        if "password" in st.session_state and st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store in session state
        elif "password" in st.session_state:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input
        st.text_input(
            "Enter App Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input(
            "Enter App Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()  # Do not run the rest of the app if password is not correct

# --- Sidebar: Config & Keys ---
with st.sidebar:
    st.header("🔑 Configuration")
    
    # --- Google API Key ---
    default_google_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
    google_api_key = st.text_input("Google API Key", value=default_google_key, type="password", help="Required for Gemini Models")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # --- Groq API Key ---
    default_groq_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    groq_api_key = st.text_input("Groq API Key (Optional)", value=default_groq_key, type="password", help="Required for Llama3/Mixtral (free at console.groq.com)")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    st.markdown("---")
    st.header("🧠 Model Selection")
    
    model_options = {
        "Gemini 2.0 Flash (Stable)": "models/gemini-2.0-flash",
        "Gemini 2.0 Flash Lite (Fast)": "models/gemini-2.0-flash-lite-preview-02-05",
        "Gemini 3 Pro Preview (Powerful)": "models/gemini-3-pro-preview",
        "Groq: Llama 3.3 70B": "llama-3.3-70b-versatile",
        "Groq: Llama 3.1 8B (Instant)": "llama-3.1-8b-instant",
        "Groq: Mixtral 8x7b": "mixtral-8x7b-32768",
    }
    
    selected_label = st.selectbox(
        "Choose LLM:", 
        list(model_options.keys()), 
        index=0
    )
    selected_model = model_options[selected_label]
    
    # Validation
    if "Groq" in selected_label and not groq_api_key:
        st.error("⚠️ Groq API Key required for this model!")
        st.stop()
    if "Gemini" in selected_label and not google_api_key:
        st.warning("⚠️ Google API Key required for this model!")
        st.stop()

    if google_api_key or groq_api_key:
        st.success("API Keys Configured!")

    st.markdown("---")
    st.header("📂 Data Source")
    source_choice = st.radio("Choose Source:", ["📄 Upload Files", "☁️ Google Drive (Coming Soon)"])

# --- Model Setup (Cached) ---
@st.cache_resource
def setup_models(model_id):
    # Dynamic Model Selection
    if "llama" in model_id or "mixtral" in model_id or "gemma2" in model_id:
        llm = Groq(model=model_id, temperature=0.5)
    else:
        llm = Gemini(model=model_id, temperature=0.5)
        
    # Use Local Embeddings to avoid "ResourceExhausted" API errors
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embed_model

llm, embed_model = setup_models(selected_model)
Settings.llm = llm
Settings.embed_model = embed_model

# --- Main Interface ---
st.title("🔒 PrivateRAG")
st.caption("Secure, Cloud-Hosted RAG powered by Gemini 1.5 Flash")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Upload a document to start chatting."}]
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

# --- Ingestion Logic ---
if source_choice == "📄 Upload Files":
    uploaded_files = st.file_uploader("Upload PDF, TXT or MD files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing files..."):
            # Create a temporary directory to store uploaded files for LlamaIndex to read
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Load data
                documents = SimpleDirectoryReader(temp_dir).load_data()
                
                # Index data (Ephemeral / In-Memory)
                index = VectorStoreIndex.from_documents(documents)
                
                # Create Query Engine
                st.session_state.query_engine = index.as_query_engine()
                st.toast(f"Indexed {len(documents)} pages!", icon="✅")

elif source_choice == "☁️ Google Drive (Coming Soon)":
    st.info("Google Drive integration is planned for the next phase. Please use File Upload for now.")

# --- Chat Logic ---
if st.session_state.query_engine:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.query_engine.query(prompt)
                st.markdown(response.response)
                
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response.response})

else:
    if source_choice == "📄 Upload Files" and not uploaded_files:
        st.info("👆 Please upload a file to begin.")
