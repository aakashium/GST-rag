# GST RAG Application

A Retrieval-Augmented Generation (RAG) application for answering questions about GST (Goods and Services Tax) laws using AI.

## Features

- 📚 **Semantic Search**: Intelligent retrieval from GST Acts and Rules PDFs
- 🤖 **AI-Powered Answers**: Uses Google Gemini (gemini-2.0-flash) for natural language responses
- ⏱️ **Rate Limiting**: Built-in delays to respect Gemini API free‑tier quotas
- 💬 **Chat Interface**: User‑friendly Streamlit‑based UI
- 📊 **Evaluation Metrics**: Automated quality assessment (results saved in `evaluation_results/`)
- 🔓 **Open Source Stack**: ChromaDB + HuggingFace embeddings

## Quick Start

### 1. Setup Environment
```cmd
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Ingest Data
```cmd
venv\Scripts\python ingest.py
```

### 4. Run Application
```cmd
venv\Scripts\streamlit run app.py
```

Visit `http://localhost:8501` to use the app.

## Project Structure

```
GST-rag/
├── Data/                   # PDF documents (GST Acts and Rules)
├── vectorstore/            # ChromaDB vector database
├── venv/                   # Python virtual environment
├── ingest.py              # Data processing script
├── rag_engine.py          # RAG chain logic
├── app.py                 # Streamlit frontend
├── evaluate.py            # Evaluation script
├── requirements.txt       # Python dependencies
└── .env                   # API key configuration
```

## Technology Stack

- **LLM**: Google Gemini (gemini-2.0-flash)
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: ChromaDB
- **Framework**: LangChain
- **Frontend**: Streamlit
- **Language**: Python 3.13

## Usage

### Ask Questions
Once the app is running, you can ask questions like:
- "What is the penalty for not filing GST returns?"
- "Explain the composition scheme."
- "What are the rules for input tax credit?"

### Run Evaluation
```cmd
venv\Scripts\python evaluate.py
# Note: The script includes built‑in delays (≈60 s) to stay within Gemini free‑tier limits.
# Evaluation results are saved under `evaluation_results/`.
```

## License

MIT License