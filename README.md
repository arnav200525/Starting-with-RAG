# Starting with RAG 🤖

This repository documents my journey into building **Retrieval-Augmented Generation (RAG)** systems. It covers the full pipeline: from raw document ingestion and vectorization to context-aware conversational AI.

## 📖 About the Project
I built this project to learn how to provide Large Language Models (LLMs) with specific, external data. These scripts demonstrate how to process a knowledge base—including historical and technical data on **Tea**, **Coffee**, and **Milk**—and use it to provide factually grounded answers.

## 🛠️ Technical Features
- **Data Ingestion**: A pipeline in `ingestion.py` that loads `.txt` files using `DirectoryLoader`, splits them into optimized chunks using `CharacterTextSplitter`, and populates a vector store.
- **Vector Database**: Utilizes **ChromaDB** with a cosine similarity metric to store and retrieve document embeddings.
- **Embeddings**: Employs HuggingFace's `all-MiniLM-L6-v2` model to transform text into high-dimensional searchable vectors.
- **Dual LLM Orchestration**:
    - **Google Gemini**: Implemented in `retrieval.py` for high-performance cloud-based inference.
    - **Ollama (Phi-3)**: Implemented in `retrieval_ollama.py` for fully local and private RAG execution.
- **History-Aware Generation**: The `history_aware_generation.py` script uses conversational memory to rewrite user queries into standalone search terms, ensuring the AI maintains context over multiple turns.

## 📂 File Overview
- `ingestion.py`: The starting point. Processes the `source/` directory to build the `chroma_db`.
- `retrieval.py`: Basic RAG implementation using the Google Gemini API.
- `retrieval_ollama.py`: Local RAG implementation using Ollama's `phi3` model.
- `history_aware_generation.py`: Advanced script managing chat history and standalone query generation.
- `source/`: Contains the text data sources (e.g., `Tea.txt`, `Green_tea.txt`, `Coffee.txt`, `Milk.txt`).

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
