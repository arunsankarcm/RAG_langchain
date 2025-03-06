# RAG with LangChain

This repository contains a Retrieval Augmented Generation (RAG) implementation using LangChain. The system allows you to query PDF documents using Mistral-7B LLM through HuggingFace's API.

## Project Structure

- `create_memory_for_llm.py`: Creates vector embeddings from PDF documents using FAISS
- `connect_memory_with_llm.py`: Connects the vector store with Mistral LLM for answering queries
- `data/`: Directory containing PDF documents to be processed
- `vectorstore/`: Directory containing FAISS vector embeddings

## Setup

1. Install dependencies:
```bash
pipenv install
```

2. Set up environment variables:
Create a `.env` file with:
```
HF_TOKEN=your_huggingface_token
```

3. Add PDF documents to the `data/` directory

## Usage

1. First, create the vector store:
```bash
python create_memory_for_llm.py
```

2. Then, query your documents:
```bash
python connect_memory_with_llm.py
```

## Technical Details

- Uses FAISS for vector storage
- Employs Mistral-7B-Instruct-v0.3 as the LLM
- Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings
- Implements chunk size of 500 with 50 token overlap for document splitting 