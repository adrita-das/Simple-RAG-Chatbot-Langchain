# Chat with Docs 📚
> A Retrieval-Augmented Generation (RAG) application that lets you chat with your PDF documents and web pages using Cohere embeddings, Qdrant vector search, and reranking.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)
![Cohere](https://img.shields.io/badge/Cohere-AI-coral)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-purple)

---

## What it does

Upload any PDF or paste a web URL, then ask questions about the content in plain English. The app retrieves the most relevant passages, reranks them for accuracy, and generates a grounded answer with citations showing exactly which page the information came from.

**Key features:**
- Upload multiple PDF files and/or a web URL as knowledge sources
- Multi-turn conversation — ask follow-up questions naturally
- Cohere reranking for higher-quality answers (not just cosine similarity)
- Source citations with real PDF page numbers
- Fully local vector storage — no cloud database required

---

## How it works

```
User question
      ↓
Embed question (Cohere embed-english-v3.0)
      ↓
Retrieve top 15 chunks (Qdrant local in-memory)
      ↓
Rerank to top 4 (Cohere rerank-v4.0-pro)
      ↓
Generate answer with citations (Cohere command-r-08-2024)
      ↓
Display answer + source page numbers
```

---

## Tech stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| Embeddings | Cohere `embed-english-v3.0` |
| Vector store | Qdrant (local in-memory mode) |
| Reranking | Cohere `rerank-v4.0-pro` |
| LLM | Cohere `command-r-08-2024` |
| PDF parsing | PyPDF |
| Web loading | LangChain WebBaseLoader |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Conversation | LangChain ConversationalRetrievalChain |

---

## Getting started

### Prerequisites
- Python 3.10+
- A free [Cohere API key](https://cohere.com) (no credit card required)

### Installation

```bash
# Clone the repo
git clone https://github.com/adrita-das/rag-streamlit-app.git
cd rag-streamlit-app

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
COHERE_API_KEY=your_cohere_api_key_here
```

### Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Usage

1. **Upload documents** — use the sidebar to upload one or more PDFs, or paste a web URL
2. **Click Process** — the app chunks, embeds, and indexes your documents
3. **Ask questions** — type in the chat box and press Enter
4. **View sources** — expand the Sources section under each answer to see which pages were cited

---

## Project structure

```
rag-streamlit-app/
├── main.py              # main Streamlit application
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── .gitignore
└── README.md
```

---

## What I learned building this

- How RAG systems work end-to-end: chunking → embedding → retrieval → generation
- Why reranking improves answer quality over raw cosine similarity
- How Cohere's citation API attributes specific claims to source documents
- Managing Streamlit session state for multi-turn conversation
- LangChain document loaders and text splitters for maintaining page metadata through the pipeline

---



