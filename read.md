# Finance News Research Tool

Ask questions about financial news articles and get answers grounded in the source, with citations.

## Features
- Paste up to 3 news article URLs; the app fetches, chunks and indexes them
- Semantic search with Hugging Face `all-MiniLM-L6-v2` embeddings + FAISS
- Chat interface with follow-up questions (follow-ups are rewritten into standalone search queries)
- Streaming answers from `gpt-oss-20b` served via Groq
- Inline citations [1], [2] linked to the exact excerpts used

## Tech stack
Python · Streamlit · LangChain · FAISS · Hugging Face · Groq (gpt-oss-20b)

## Run locally
```bash
pip install -r requirements.txt
cp .env.example notebooks/.env   # then add your Groq API key
streamlit run notebooks/main.py
```

## Roadmap
- [ ] Hybrid search (BM25 keyword + semantic, merged with Reciprocal Rank Fusion)
- [ ] Retrieval evaluation (Hit Rate, MRR)
- [ ] Live demo on Streamlit Community Cloud

## Credits
Started from a YouTube tutorial and extended with chat, follow-up rewriting, streaming and citations.