import streamlit as st
import os
import tempfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import json
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.google import GoogleDriveReader

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
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

# --- Data Source Selection ---
st.markdown("---")
st.header("📂 Data Source")
source_choice = st.radio("Select Input Method:", ["📄 Upload Files", "☁️ Google Drive (Service Account)"], horizontal=True)

documents = []

# --- Ingestion Logic ---
if source_choice == "📄 Upload Files":
    uploaded_files = st.file_uploader("Upload PDF, TXT or MD files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing files..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                reader = SimpleDirectoryReader(input_dir=temp_dir)
                documents = reader.load_data()
                st.toast(f"✅ Loaded {len(documents)} pages from files!", icon="📄")

elif source_choice == "☁️ Google Drive (Service Account)":
    st.info("ℹ️ Requires a Google Cloud Service Account shared with the target folder.")
    folder_id_input = st.text_input("Enter Google Drive Folder ID:", help="The ID is the last part of the folder URL.")
    folder_id = folder_id_input.strip() if folder_id_input else ""
    
    if st.button("Ingest from Drive"):
        if "google_drive" not in st.secrets:
            st.error("❌ `google_drive` section missing in secrets.toml!")
        elif not folder_id:
            st.warning("⚠️ Please enter a Folder ID.")
        else:
            with st.spinner("Connecting to Google Drive..."):
                try:
                    # Create temp credentials file from secrets
                    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp_creds:
                        json.dump(dict(st.secrets["google_drive"]), temp_creds)
                        temp_creds.flush()
                        creds_path = temp_creds.name
                    
                    # Load data
                    loader = GoogleDriveReader(service_account_key_path=creds_path)
                    loaded_docs = loader.load_data(folder_id=folder_id)
                    
                    # Safe assignment
                    if loaded_docs:
                        documents = loaded_docs
                        st.toast(f"✅ Loaded {len(documents)} docs from Drive!", icon="☁️")
                    else:
                        st.warning("⚠️ No documents found in this folder. Check permissions or folder ID.")
                    
                    # Cleanup
                    os.unlink(creds_path)
                    
                except Exception as e:
                    st.error(f"Error accessing Drive: {str(e)}")
                    # Cleanup on error
                    if 'creds_path' in locals() and os.path.exists(creds_path):
                        os.unlink(creds_path)

# --- Indexing & Engine Construction ---
# If new documents were ingested, update the index
if documents:
    with st.spinner("Indexing documents..."):
        st.session_state.vector_index = VectorStoreIndex.from_documents(documents)
        st.toast("✅ Indexing Complete!", icon="🧠")

# Always rebuild the query engine from the current index and current LLM
if st.session_state.vector_index:
    st.session_state.query_engine = st.session_state.vector_index.as_query_engine(llm=llm)
else:
    st.session_state.query_engine = None

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
