# Finance News Research Tool

Ask questions about financial news articles and get answers grounded in the source, with citations.

## Features
- Paste up to 3 news article URLs; the app fetches, chunks and indexes them
- Hybrid search: BM25 keyword search + semantic search (Hugging Face `all-MiniLM-L6-v2` embeddings + FAISS), merged with Reciprocal Rank Fusion
- Search mode switch to compare Hybrid, Meaning only and Keywords only
- Chat interface with follow-up questions (follow-ups are rewritten into standalone search queries)
- Streaming answers from `gpt-oss-20b` served via Groq
- Inline citations [1], [2] linked to the exact excerpts used

## How hybrid search works
- Keyword search (BM25) finds exact terms like tickers, acronyms and figures; semantic search finds matching meaning even when the words differ ("bad loans" vs "NPA")
- Each search returns its top 10 chunks, and Reciprocal Rank Fusion merges them by rank position
- Common filler words are ignored in keyword search, so they don't create junk matches
- At most 2 chunks per article make it into the final 4, so answers can draw on several articles

## Tech stack
Python · Streamlit · LangChain · FAISS · Hugging Face · rank-bm25 · Groq (gpt-oss-20b)

## Run locally
```bash
pip install -r requirements.txt
cp .env.example notebooks/.env   # then add your Groq API key
streamlit run notebooks/main.py
```

## Roadmap
- [x] Hybrid search (BM25 keyword + semantic, merged with Reciprocal Rank Fusion)
- [ ] Retrieval evaluation (Hit Rate, MRR)
- [ ] Live demo on Streamlit Community Cloud

## Credits
Started from a YouTube tutorial and extended with chat, follow-up rewriting, streaming, citations and hybrid search.
