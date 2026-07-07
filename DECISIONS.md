# Technical Decisions Log

This document records key architectural and implementation decisions for the AI Ethics Assistant project, including the rationale behind each choice and alternatives considered.

## Domain & Knowledge Base

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 1 | Domain: European AI ethics | Payroll, clinical data, ML interview prep | Public value; rich official documentation; personally motivated; addresses critical regulatory gap |
| 2 | 4 documents (~250 pages) | Minimum 3 docs / 20 pages | Depth: law + technical + principles + case studies; comprehensive coverage without overwhelming complexity |
| 3 | Document diversity (legal, academic, policy) | Single source type | Multiple perspectives: legal requirements, technical challenges, real-world cases, ethical principles |

## RAG Configuration

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 4 | chunk_size=1400, overlap=150 | 500/50, 1000/150, 2000/200 | Legal provisions need larger coherent chunks (verified via retrieval tests); 1400 produces ~750 chunks for efficient retrieval; maintains context without excessive fragmentation |
| 5 | TOP_K=4 | 2, 8 | 4 gives multi-source context without noise; balances breadth and relevance; tested empirically for best results |
| 6 | temperature=0.2 | 0.0, 0.7 | Factual domain → low creativity; not 0 to keep fluent phrasing; reduces hallucination risk while maintaining readability |

## Technology Stack

### LLM Selection

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 7 | Groq LLM (llama-3.1-8b-instant) | Gemini, OpenAI, Claude | Free, no rate limits, fast inference; suitable for educational project; Llama 3.1 provides excellent quality; eliminates cost barriers |

**Detailed Rationale:**
- **Gemini API**: Initially used but had strict rate limits (1000 requests/day for embeddings) that blocked our use case; also had daily quotas that prevented experimentation
- **OpenAI API**: Requires payment, has rate limits, not suitable for educational projects with zero budget
- **Claude API**: Expensive, not suitable for educational projects; requires API key with payment setup
- **Groq**: Completely free, no usage limits, sub-second response times, high-quality Llama 3.1 model, perfect for learning and demonstration

### Embeddings

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 8 | Sentence Transformers (local) | OpenAI embeddings, Gemini embeddings | Zero cost, no API calls, no rate limits; runs locally; all-MiniLM-L6-v2 provides excellent semantic understanding; processes 758 chunks in ~13 seconds |

**Detailed Rationale:**
- **OpenAI Embeddings**: Costs money, has rate limits, not suitable for zero-budget educational project
- **Gemini Embeddings**: Had strict daily quotas (1000 requests/day), blocked our use case during development
- **Cohere Embeddings**: Not free for production use, requires API key
- **Sentence Transformers**: Completely free, runs locally (no API calls), no rate limits, high quality, privacy-preserving (no data sent externally)

### Vector Database

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 9 | ChromaDB vector store | Pinecone, Weaviate, FAISS | Open-source, local deployment, no external dependencies; excellent LangChain integration; efficient similarity search; persistent storage |

**Detailed Rationale:**
- **Pinecone**: Requires cloud account, has usage limits, not suitable for local development
- **Weaviate**: More complex setup, cloud-focused, steeper learning curve
- **FAISS**: Less user-friendly, no built-in persistence, requires additional storage management
- **ChromaDB**: Open-source, local deployment, excellent LangChain support, persistent storage, efficient search, easy setup

### Agent Framework

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 10 | LangGraph with MemorySaver | LangChain Agents, Custom implementation | Modern architecture, built-in memory management, flexible workflow definition, type safety with TypedDict, seamless LangChain integration |

**Detailed Rationale:**
- **LangChain Agents**: Older architecture, less flexible, more complex state management
- **Custom Implementation**: More complex, less maintainable, requires building memory management from scratch
- **LangGraph**: State-of-the-art agent framework, built-in MemorySaver for conversation context, flexible node-based architecture, strong typing, excellent documentation

### Web Interface

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 11 | Streamlit | Flask/Django, React, Gradio | Python-native, fast development, built-in chat components, one-click deployment to Streamlit Cloud, free hosting |

**Detailed Rationale:**
- **Flask/Django**: Requires more development time, need to build UI from scratch, more complex deployment
- **React**: Requires frontend development skills, separate backend/frontend, more complex deployment
- **Gradio**: Less flexible for custom UIs, limited customization options
- **Streamlit**: Python-native (no frontend skills needed), built-in chat interface, rapid development, free hosting on Streamlit Cloud, excellent for ML/AI demos

## System Design

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 12 | "I don't know" rule in system prompt | Let model answer freely | Prevents hallucinated legal claims — core responsible-AI choice; ensures accuracy in legal/ethics domain |
| 13 | Source attribution in responses | No source attribution | Enables verification; shows retrieval is working; builds trust; educational value |
| 14 | Conversation memory with MemorySaver | Stateless responses | Maintains context across questions; enables follow-up queries; better user experience |
| 15 | Legal disclaimer in system prompt | No disclaimer | Responsible AI practice; especially important for legal domain; protects users from relying on AI for legal decisions |
| 28 | Supervisor/verification node (generate → verify → revise) | Single-pass generation only; self-critique in one prompt; full multi-agent debate | Testing showed the single-pass agent occasionally fabricated a plausible citation (e.g. an EU AI Act article number) not present in the retrieved context, especially on vague multi-turn follow-ups. A second, independent LLM call re-checks the draft against the same context and forces one regeneration if it finds an unsupported claim. Cheaper and simpler than a full multi-agent debate — one extra Groq call in the common case, two if a correction is needed — while still being a real generator+critic pattern rather than the same model grading its own homework in one shot. Especially important given this is a compliance/ethics assistant, where a confident but ungrounded answer is worse than "I don't know." |

## Deployment Strategy

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 16 | Include chroma_db in repository | Generate on deployment | Simplifies deployment; no need for build step on cloud; ensures consistency; faster cold starts |
| 17 | Streamlit Cloud deployment | Self-hosted, other cloud platforms | Free hosting; easy deployment; automatic scaling; no infrastructure management |
| 18 | Environment variables for API keys | Hardcoded keys | Security best practice; prevents accidental exposure; enables different keys for different environments |

## Performance Optimization

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 19 | Batch processing for embeddings | Single document processing | More efficient; reduces overhead; faster overall processing |
| 20 | Local embeddings model | API-based embeddings | Zero latency; no rate limits; no costs; better privacy |
| 21 | Caching embeddings in ChromaDB | Re-compute on each query | Dramatically faster response times; reduced computational overhead |

## Security & Privacy

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 22 | .env file for API keys | Hardcoded in code | Security best practice; prevents accidental commits; enables different environments |
| 23 | .gitignore for sensitive files | Commit all files | Prevents accidental exposure of secrets; protects API keys and local data |
| 24 | Local embeddings (no data sent externally) | Cloud-based embeddings | Privacy-preserving; no data leaves user's machine; GDPR compliant |

## Development Workflow

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 25 | Separate build script (build_vector_store.py) | Include in notebook | Reusable; easier to run; better separation of concerns; can be automated |
| 26 | Type hints in code | No type hints | Better code quality; IDE support; easier maintenance; catches errors early |
| 27 | Modular code structure | Monolithic script | Easier to test; better maintainability; clearer code organization |

## Lessons Learned

### What Worked Well
1. **Free Stack Choice**: Groq + local embeddings eliminated all cost barriers and rate limits
2. **RAG Architecture**: Provided grounded, verifiable answers with source attribution
3. **Streamlit**: Enabled rapid development and easy deployment without frontend skills
4. **ChromaDB**: Simple, efficient vector storage with excellent LangChain integration

### What We'd Do Differently
1. **Earlier Migration to Free Stack**: Should have started with Groq instead of Gemini to avoid rate limit issues
2. **Vector Store in Repository**: Should have included chroma_db from the beginning for easier deployment
3. **More Testing**: Could have tested different chunk sizes and TOP-K values more systematically

### Key Takeaways
1. **Rate Limits Matter**: Free tiers with quotas can block development; choose truly free alternatives
2. **Local Processing**: Running embeddings locally eliminates many API-related issues
3. **Deployment Planning**: Consider deployment requirements from the start (e.g., vector store storage)
4. **Documentation**: Comprehensive documentation helps with future maintenance and onboarding |
