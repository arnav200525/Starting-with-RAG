# Chatbot Project

This folder contains a Retrieval-Augmented Generation (RAG) chatbot workflow for answering questions using local document knowledge.

## Overview

The project is designed to:
- load text documents from a local source folder
- split them into smaller chunks for better retrieval
- create embeddings and store them in a vector database
- retrieve relevant document chunks for a user question
- generate a grounded answer using a language model
- keep chat context for follow-up questions when needed

## Folder Structure

Core project files:
- `general_chat.py` - simple chat interface for direct conversational use
- `history_aware_generation.py` - question-answering flow with follow-up handling and retrieval
- `ingestion.py` - document loading, chunking, and vector store creation
- `requirements.txt` - project dependencies
- `Source/` - source documents used for retrieval

Git-ignored files:
- `db/` - persistent vector database storage (generated and ignored by Git)
- `.env` - environment configuration for API keys (typically kept out of version control)
- `.venv/` - local virtual environment folder (ignored by Git)

## Prerequisites

Before running the project, make sure you have:
- Python 3.10 or newer
- A virtual environment activated
- A valid Groq API key stored in the environment

Example environment variable:

```bash
GROQ_API_KEY=your_api_key_here
```

## Setup

1. Open the project folder.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add your source documents in the `Source/` folder.
5. Make sure the `.env` file contains your API key.

## How It Works

1. Documents are loaded from the source directory.
2. The text is split into manageable chunks.
3. Embeddings are created for each chunk.
4. The vector database stores these embeddings.
5. When a user asks a question, relevant chunks are retrieved.
6. The model uses those chunks to craft a grounded response.

## Running the Chatbot

You can run the interactive chat script as follows:

```bash
python general_chat.py
```

For the retrieval-based Q&A flow:

```bash
python history_aware_generation.py
```

## Notes

- The system relies on local documents for answer generation.
- Retrieval quality depends on the quality and structure of your source files.
- You can add more documents to the source directory and rebuild the vector store when needed.

## Recommended Use

This project is suitable for:
- internal knowledge base chatbots
- document Q&A over text sources
- lightweight RAG prototypes using local files

## License

This project is intended for educational and demo use unless otherwise specified by the repository owner.
