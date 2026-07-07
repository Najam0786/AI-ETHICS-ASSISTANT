# Technical Decisions Log

This document records key architectural and implementation decisions for the AI Ethics Assistant project.

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 1 | Domain: European AI ethics | Payroll, clinical data, ML interview prep | Public value; rich official documentation; personally motivated |
| 2 | 4 documents (~250 pages) | Minimum 3 docs / 20 pages | Depth: law + technical + principles + case studies |
| 3 | chunk_size=1400, overlap=150 | 500/50, 1000/150, 2000/200 | Legal provisions need larger coherent chunks (verified via retrieval tests); 1400 produces ~750 chunks for efficient retrieval |
| 4 | TOP_K=4 | 2, 8 | 4 gives multi-source context without noise |
| 5 | temperature=0.2 | 0.0, 0.7 | Factual domain → low creativity; not 0 to keep fluent phrasing |
| 6 | MemorySaver checkpointer | Manual history list | Native LangGraph pattern; thread-scoped; less code |
| 7 | "I don't know" rule in system prompt | Let model answer freely | Prevents hallucinated legal claims — core responsible-AI choice |
| 8 | Groq LLM (llama-3.1-8b-instant) | Gemini, OpenAI, Claude | Free, no rate limits, fast inference; suitable for educational project |
| 9 | Sentence Transformers (local) | OpenAI embeddings, Gemini embeddings | Zero cost, no API calls, no rate limits; runs locally |
| 10 | ChromaDB vector store | Pinecone, Weaviate, FAISS | Open-source, local deployment, no external dependencies | |
