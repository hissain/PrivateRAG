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
- **Google Cloud Service Account** (for Drive Integration)

### Google Drive Setup (Important)

To use the Google Drive integration, you must:

1.  **Enable the API**: Go to [Google Cloud Console](https://console.developers.google.com/apis/api/drive.googleapis.com/overview) and **ENABLE** the "Google Drive API" for your project.
2.  **Create Service Account**: 
    - Go to "IAM & Admin" > "Service Accounts".
    - Create a new account and download the **JSON Key**.
3.  **Share Folders**:
    - You **MUST** explicitly share your Google Drive folders with the Service Account Email (e.g., `driverag@...iam.gserviceaccount.com`).
    - Give it **Viewer** permission.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/hissain/PrivateRAG.git
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

    [google_drive]
    # Paste your Service Account JSON content here (keys/values)
    type = "service_account"
    project_id = "your_project_id"
    private_key_id = "your_private_key_id"
    private_key = "-----BEGIN PRIVATE KEY-----\n..."
    client_email = "your_service_account_email@..."
    client_id = "..."
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "..."
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
    - **Gemini 3 Pro**: Recommended for reasoning.
3.  **Data Ingestion**:
    - **Upload Files**: Drag and drop PDF, TXT, or MD files.
    - **Google Drive**: Select "Google Drive", enter the **Folder ID** (from your browser URL), and click "Ingest".
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
