# Technical Decisions Log

| # | Decision | Alternatives considered | Rationale |
|---|---|---|---|
| 1 | Domain: European AI ethics | Payroll, clinical data, ML interview prep | Public value; rich official documentation; personally motivated |
| 2 | 4 documents (~250 pages) | Minimum 3 docs / 20 pages | Depth: law + technical + principles + case studies |
| 3 | chunk_size=1400, overlap=150 | 500/50, 1000/150, 2000/200 | Legal provisions need larger coherent chunks (verified via retrieval tests); 1400 also keeps total chunks (~750) under Gemini's free-tier daily embedding quota (1000/day) for ~250 pages — at 1000/150 the same corpus produced 1043 chunks, already over the cap |
| 4 | TOP_K=4 | 2, 8 | 4 gives multi-source context without noise |
| 5 | temperature=0.2 | 0.0, 0.7 | Factual domain → low creativity; not 0 to keep fluent phrasing |
| 6 | MemorySaver checkpointer | Manual history list | Native LangGraph pattern; thread-scoped; less code |
| 7 | "I don't know" rule in system prompt | Let model answer freely | Prevents hallucinated legal claims — core responsible-AI choice |
| 8 | Batched indexing (90 chunks/batch, 61s pause) | Single `from_documents()` call | Gemini free tier caps embedding calls at 100 requests/minute; a single bulk call blew through the quota mid-run (`ResourceExhausted`) |
