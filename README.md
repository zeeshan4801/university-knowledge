# university-knowledge
# 🎓 University Academic Knowledge Assistant

A Retrieval-Augmented Generation (RAG) based academic assistant built using:

- Streamlit
- FAISS Vector Database
- HuggingFace Embeddings
- Groq LLM


## Features

✅ Multi-document academic search

✅ Source traceability

✅ Page-level citations

✅ Fast semantic retrieval

✅ Cloud deployment ready


## Architecture


PDF Documents

↓

Embeddings

↓

FAISS Vector Database

↓

Retriever

↓

Groq LLM

↓

Answer with Sources



## Vector Database

The repository contains:

- FAISS index
- Metadata JSON

Original PDF documents are not uploaded for privacy and size reasons.


## Deployment

1. Upload repository to GitHub

2. Add GROQ_API_KEY in Streamlit secrets

3. Deploy using Streamlit Cloud


## Model Information

Embedding:

BAAI/bge-base-en-v1.5


LLM:

openai/gpt-oss-120b


## Source Citation

Every response includes:

- Document name
- Page number
- Source reference
