# PrivateRAG

PrivateRAG is a secure, cloud-hosted Retrieval Augmented Generation (RAG) application. It allows users to interact with sensitive documents (PDF, TXT, MD) using LLMs (Gemini, Groq) without exposing data to public training sets. The architecture prioritizes privacy by using in-memory processing and app-level security.

## Features

- **Multi-Provider Support**: Dynamic switching between Google Gemini (Stable/Pro) and Groq (Llama 3/Mixtral).
- **Privacy-First**:
    - **In-Memory Store**: Documents are processed in RAM and discarded after the session.
    - **App-Level Password**: Gated access to prevent unauthorized usage.
    - **Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` on CPU (no external embedding API calls).
- **Zero Cost**: Optimized for Streamlit Community Cloud free tier.

## Tech Stack

- **Framework**: Streamlit
- **RAG Engine**: LlamaIndex
- **LLMs**: Google Gemini 2.0/3.0, Groq Llama 3.3 / Mixtral
- **Embeddings**: HuggingFace (Local)

## Setup and Installation

### Prerequisites

- Python 3.9+
- Google API Key (for Gemini)
- Groq API Key (for Llama/Mixtral)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/PrivateRAG.git
    cd PrivateRAG
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Configure Secrets (Local):
    Create `.streamlit/secrets.toml`:
    ```toml
    APP_PASSWORD = "your_secure_password"
    GOOGLE_API_KEY = "your_google_key"
    GROQ_API_KEY = "your_groq_key"
    ```

4.  Run the application:
    ```bash
    streamlit run app.py
    ```

## Usage Guide

1.  **Login**: Enter the `APP_PASSWORD`.
2.  **Select Model**: Use the sidebar to choose the LLM.
    - **Groq Llama 3.3**: Recommended for speed.
    - **Gemini 3 Pro**: Recommended for reasoning.
3.  **Upload**: Drag and drop PDF, TXT, or MD files.
4.  **Chat**: Ask questions about the uploaded content.

## Deployment

Deploy to **Streamlit Community Cloud**:

1.  Push code to GitHub.
2.  Connect repository in Streamlit Cloud.
3.  Navigate to **App Settings > Secrets** and paste your API keys and Password.
4.  Deploy.

## Troubleshooting

### Deployment Fails?
- Check `requirements.txt` is present.
- Ensure python version is 3.9+.

### Model Errors (429/404)?
- Switch providers using the dropdown (e.g., from Gemini to Groq).
- Verify API Keys in Secrets.

### Upload Issues?
- Ensure files are <50MB.
- Supported text formats: PDF, TXT, MD.

## Security

- **Secrets**: Managed via `secrets.toml` or Cloud Secrets. Never committed.
- **Data**: Ephemeral (RAM only). destroyed on session restart.
