# Architecture Documentation

This document provides detailed technical architecture information for the AI Ethics Assistant project.

## System Overview

The AI Ethics Assistant is a RAG-based (Retrieval-Augmented Generation) conversational agent that provides accurate, grounded answers about European AI ethics and regulation. The system combines vector similarity search with large language model generation to ensure responses are based on authoritative sources.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Interface                                   │
│                      (Streamlit Web App)                                 │
└────────────────────────┬──────────────────────────────────────────────-─┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          LangGraph Agent                                 │
│                                                                           │
│  ┌──────────┐   low score   ┌─────────┐                                 │
│  │ Retrieve │──────────────▶│ Abstain │───────────────────┐             │
│  │ (MMR +   │               └─────────┘                   │             │
│  │  score)  │  confident                                   ▼             │
│  └────┬─────┘─────────▶┌──────────┐   ┌─────────┐   ┌──────────┐        │
│       │                │ Generate │──▶│ Verify  │──▶│ Finalize │──▶ END │
│       ▼                └──────────┘   └────┬────┘   └────▲─────┘        │
│  ┌──────────┐                          invalid │            │            │
│  │ ChromaDB │                               ▼               │            │
│  │  Vector  │                         ┌─────────┐           │            │
│  │  Store   │                         │ Revise  │───────────┘            │
│  └──────────┘         Generate/Verify/Revise all call Groq LLM           │
└─────────────────────────────────────────────────────────────────────────┘
```

Four independent checks, not one — see [Hallucination Defenses](#hallucination-defenses) below for why each layer exists and what it catches:

- **`Retrieve` → `Abstain`** (deterministic, no LLM call): if the best-matching chunk's similarity score exceeds a calibrated threshold, the agent refuses immediately rather than trusting the LLM's judgment about whether it knows enough.
- **`Retrieve` uses MMR**, not plain top-k similarity, so the 4 chunks handed to the LLM are diverse rather than near-duplicates of each other.
- **`Verify`** runs a deterministic citation cross-check first (does the draft cite a source that wasn't actually retrieved?), then — only if that passes — a second, independent Groq call checking both faithfulness (is every claim supported by context?) and relevance (does the draft actually answer the question, rather than dodging it?).
- **`Revise`** regenerates once, told exactly what check failed and why, then always routes to `Finalize`.

This generator+critic pattern, combined with deterministic gates that don't depend on any LLM's self-judgment, catches fabricated citations, off-scope questions, and evasive-but-technically-grounded answers that a single-pass RAG agent would let through — which matters for a compliance/ethics domain where a confident, ungrounded answer is worse than "I don't know."

## Hallucination Defenses

| # | Defense | Node(s) | Type | Catches |
|---|---|---|---|---|
| 1 | Retrieval confidence gate | `retrieve` → `abstain` | Deterministic | Out-of-scope questions, before any generation happens |
| 2 | MMR retrieval | `retrieve` | Retrieval quality | Near-duplicate context that starves the LLM of real coverage |
| 3 | Citation cross-check | `verify` | Deterministic | A cited source that was never retrieved for this query |
| 4 | Faithfulness + relevance check | `verify` → `revise` | LLM (independent call) | Fabricated claims, and evasive answers that dodge the actual question |

**Threshold calibration (layer 1):** Chroma's default distance is squared L2 (lower = more similar). Testing this corpus with `all-MiniLM-L6-v2` embeddings found in-scope questions scored 0.43–0.75 and out-of-scope questions scored 1.02–1.78 — a clean separation. `SIMILARITY_THRESHOLD = 0.9` sits in that gap with margin on both sides. A follow-up question with no self-contained meaning (e.g. "give an example of it") would otherwise always score poorly and be wrongly abstained — `retrieve` works around this by prepending the prior assistant turn to the retrieval query (but not to the question passed to `generate`), which is enough context to resolve the reference without an extra LLM call.

**Why not a single check:** an LLM checking its own (or a sibling model's) output is useful but not infallible — testing showed the generator and a naive single verifier could agree on a wrong citation. Layers 1 and 3 are rule-based and don't depend on any model's agreement, so they act as a hard floor under the LLM-based layers 2 and 4.

Every retrieval score and verification verdict is logged via Python's `logging` module, and the Streamlit UI tracks a per-session count of verified/revised/abstained queries — turning "we think hallucinations are rare" into a measured rate.

## Component Architecture

### 1. Document Processing Pipeline

**Purpose:** Convert raw PDF documents into searchable vector embeddings

**Flow:**
```
PDF Files → Text Extraction → Cleaning → Chunking → Embedding → Vector Storage
```

**Components:**
- **PyPDFLoader**: Extracts text from PDF documents
- **Text Cleaner**: Normalizes whitespace and fixes hyphenation
- **Text Splitter**: Divides text into chunks (1400 chars, 150 overlap)
- **Sentence Transformers**: Converts text to vector embeddings
- **ChromaDB**: Stores and indexes vectors for similarity search

**Implementation:** `build_vector_store.py`

### 2. Retrieval System

**Purpose:** Find relevant document chunks based on user queries, and produce a confidence signal used to decide whether to answer at all

**Flow:**
```
User Query → Embed Query → [MMR Search → Top-K Chunks] + [Top-1 Similarity Score] → Context Assembly
```

**Components:**
- **LocalEmbeddings**: Wraps Sentence Transformers for LangChain compatibility
- **ChromaDB MMR search**: Retrieves TOP_K=4 chunks balancing relevance and diversity (`fetch_k=20`, `lambda_mult=0.5`)
- **Confidence score**: A separate plain similarity search (`k=1`) used only to decide whether to abstain — MMR's re-ranked distances aren't a clean relevance signal on their own
- **Context Builder**: Assembles retrieved chunks with source attribution, and tracks which source names were actually retrieved (used by the citation check)

**Implementation:** `app/streamlit_app.py` (retrieve function)

### 3. Generation System

**Purpose:** Generate coherent responses using retrieved context

**Flow:**
```
Context + Query → System Prompt → LLM Generation → Response
```

**Components:**
- **System Prompt**: Defines agent behavior and constraints
- **ChatGroq**: Groq API client for Llama 3.1 model
- **Response Formatter**: Ensures proper citation and formatting

**Implementation:** `app/streamlit_app.py` (generate function)

### 4. Agent Framework

**Purpose:** Orchestrate retrieval and generation with conversation memory

**Flow:**
```
User Input → Agent State → Retrieve Node → Generate Node → Response
     ↑                                    │
     └──────────── Memory Update ◀────────┘
```

**Components:**
- **LangGraph StateGraph**: Defines agent workflow
- **MemorySaver**: Maintains conversation context across turns
- **AgentState**: TypedDict for state management

**Implementation:** `app/streamlit_app.py` (load_agent function)

### 5. Web Interface

**Purpose:** Provide user-friendly chat interface

**Components:**
- **Streamlit**: Web framework for Python
- **Chat Interface**: Built-in chat message components
- **Session State**: Maintains chat history
- **Error Handling**: Graceful error messages

**Implementation:** `app/streamlit_app.py` (main function)

## Data Flow

### Query Processing Flow

```
1. User enters question in Streamlit interface
2. Question added to agent state as HumanMessage
3. Agent invokes retrieve node:
   - Question embedded using Sentence Transformers
   - Similarity search in ChromaDB (TOP_K=4)
   - Top chunks assembled with source attribution
4. Agent invokes generate node:
   - System prompt formatted with retrieved context
   - LLM generates response using Groq API
   - Response added to agent state
5. Response displayed in Streamlit interface
6. Conversation state updated in MemorySaver
```

### Vector Store Creation Flow

```
1. PDF documents loaded from data/ directory
2. Text extracted and cleaned (whitespace normalization)
3. Documents split into chunks (1400 chars, 150 overlap)
4. Each chunk embedded using Sentence Transformers
5. Embeddings stored in ChromaDB with metadata
6. Vector database persisted to chroma_db/ directory
```

## Data Structures

### AgentState

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    draft: str
    verification: str
    top_score: float
    sources_retrieved: list[str]
```

**Purpose:** Maintains conversation state, retrieved context, and the scratch fields used by the abstain/verify/revise safety layers

**Fields:**
- `messages`: List of conversation messages (user + assistant) — only the *final*, checked answer is appended here, never an intermediate draft
- `context`: Retrieved document chunks for the current query
- `draft`: Candidate answer produced by `generate` (or `abstain`), pending the verify step
- `verification`: `"VALID"`, `"INVALID: <reason>"`, or `"ABSTAINED: <reason>"` — the outcome that decides routing and is logged/surfaced in the UI
- `top_score`: Confidence of the single best-matching retrieved chunk, used by the abstain gate
- `sources_retrieved`: Source names actually retrieved for this query, used by the citation cross-check

### Document Metadata

```python
{
    "source_name": "EU AI Act",  # Human-readable source name
    "page": 42,                  # Original page number
    "chunk_index": 123           # Chunk identifier
}
```

**Purpose:** Provides source attribution and traceability

### System Prompt Template

```
You are an expert assistant on European AI ethics and regulation.

Your knowledge base consists of four authoritative sources:
1. The EU AI Act (official legal text)
2. "A Survey on Bias and Fairness in Machine Learning" (Mehrabi et al.)
3. The European Parliament study "The Ethics of Artificial Intelligence: Issues and Initiatives"
4. The EU "Ethics Guidelines for Trustworthy AI" (AI HLEG)

RULES:
1. Answer ONLY using the context provided below. Never invent facts, article numbers, or legal requirements.
2. If the context does not contain the answer, say clearly: "I don't have enough information in my knowledge base to answer that"
3. Mention which source your answer is based on (e.g., "According to the EU AI Act...").
4. Be clear and educational: your users may be developers, students, or policymakers with no legal background.
5. Be concise: 2–4 short paragraphs maximum, unless the user asks for more detail.
6. You are not a lawyer. For legal decisions, recommend consulting a qualified professional.

CONTEXT FROM KNOWLEDGE BASE:
{context}
```

## Configuration

### Vector Store Configuration

```python
CHUNK_SIZE = 1400          # Characters per chunk
CHUNK_OVERLAP = 150        # Overlap between chunks
TOP_K = 4                  # Chunks retrieved per query
FETCH_K = 20               # Candidates considered by MMR before diversity re-ranking
MMR_LAMBDA = 0.5           # 0 = max diversity, 1 = pure relevance
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ai_ethics_eu"
```

### LLM Configuration

```python
LLM_MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.2                 # Low creativity for factual accuracy
SIMILARITY_THRESHOLD = 0.9        # Chroma L2 distance; above this, abstain instead of generating
```

### File Structure

```
ai-ethics-assistant/
├── app/
│   └── streamlit_app.py          # Main application
├── data/
│   ├── EU-AI-Act.pdf             # Source documents
│   ├── bias_fairness_survey.pdf
│   ├── ethics_of_ai_study.pdf
│   └── trustworthy_ai_guidelines.pdf
├── chroma_db/                     # Vector database
│   ├── chroma.sqlite3            # ChromaDB metadata
│   └── [collection-id]/          # Collection data
│       ├── data_level0.bin       # Vector embeddings
│       ├── header.bin            # Collection metadata
│       ├── index_metadata.pickle # Index information
│       ├── length.bin            # Chunk lengths
│       └── link_lists.bin        # ANN index
├── notebooks/
│   └── ai_ethics_assistant.ipynb  # Development notebook
├── build_vector_store.py          # Vector store builder
├── requirements.txt              # Dependencies
├── .env                          # Environment variables
└── .gitignore                    # Git ignore rules
```

## Performance Characteristics

### Vector Store Creation

- **Input**: 253 pages across 4 PDF documents
- **Output**: 758 chunks
- **Processing Time**: ~13 seconds
- **Memory Usage**: ~500MB (embeddings model)
- **Storage Size**: ~100MB (ChromaDB)

### Query Processing

- **Retrieval Time**: <100ms (ChromaDB similarity search)
- **Embedding Time**: <50ms (single query embedding)
- **Generation Time**: <2 seconds (Groq LLM)
- **Total Response Time**: <3 seconds

### Resource Usage

- **Embeddings Model**: 120MB (all-MiniLM-L6-v2)
- **Vector Database**: 100MB (ChromaDB)
- **Application Memory**: ~200MB (Streamlit + dependencies)
- **Total Memory**: ~420MB

## Security Considerations

### API Key Management

- **Storage**: Environment variables in `.env` file
- **Git**: `.env` excluded via `.gitignore`
- **Deployment**: Streamlit Cloud secrets for production
- **Access**: Application-level validation

### Data Privacy

- **Local Processing**: Embeddings computed locally
- **No Data Exfiltration**: No document data sent to external services
- **Query Privacy**: User queries sent only to Groq API
- **GDPR Compliance**: No personal data stored or processed

### Code Security

- **Type Hints**: Strong typing for state management
- **Input Validation**: Error handling for missing API keys and vector store
- **Dependency Management**: Version constraints in requirements.txt
- **Secrets Management**: Environment variables for sensitive data

## Scalability Considerations

### Current Limitations

- **Single User**: Designed for single-user deployment
- **Local Storage**: Vector store stored locally
- **No Caching**: Each query processes embeddings fresh
- **Memory Bound**: All data loaded into memory

### Potential Scaling Strategies

1. **Multi-User Support**:
   - Deploy ChromaDB as a service
   - Implement user-specific collections
   - Add authentication/authorization

2. **Horizontal Scaling**:
   - Containerize application
   - Use load balancer
   - Deploy multiple instances

3. **Performance Optimization**:
   - Implement query caching
   - Use GPU for embeddings
   - Optimize chunk size for faster retrieval

4. **Storage Scaling**:
   - Use cloud vector database (Pinecone, Weaviate)
   - Implement incremental updates
   - Add document versioning

## Monitoring & Observability

### Current Monitoring

- **Console Logging**: Basic progress messages
- **Error Messages**: User-facing error descriptions
- **Performance Metrics**: Manual timing measurements

### Recommended Enhancements

1. **Application Logging**:
   - Structured logging (JSON format)
   - Log levels (DEBUG, INFO, WARNING, ERROR)
   - Log aggregation (ELK stack, CloudWatch)

2. **Performance Monitoring**:
   - Query latency tracking
   - Response time percentiles
   - Error rate monitoring

3. **Usage Analytics**:
   - Query pattern analysis
   - Popular topics tracking
   - User engagement metrics

4. **Health Checks**:
   - API key validation
   - Vector store connectivity
   - LLM service availability

## Testing Strategy

### Current Testing

- **Manual Testing**: Interactive testing via Streamlit
- **Query Validation**: Manual verification of responses
- **Source Attribution**: Manual check of citations

### Recommended Testing

1. **Unit Tests**:
   - Text cleaning functions
   - Chunking logic
   - Embedding wrapper
   - Agent state management

2. **Integration Tests**:
   - Vector store creation
   - Retrieval accuracy
   - End-to-end query processing
   - Memory persistence

3. **Quality Tests**:
   - Retrieval precision/recall
   - Response relevance
   - Source attribution accuracy
   - Hallucination detection

4. **Performance Tests**:
   - Query latency benchmarks
   - Memory usage profiling
   - Concurrent request handling
   - Scalability testing

## Deployment Architecture

### Local Deployment

```
┌─────────────────────────────────┐
│   Developer Machine              │
│  ┌───────────────────────────┐  │
│  │  Python Environment       │  │
│  │  - Virtual Environment    │  │
│  │  - Dependencies           │  │
│  │  - .env file              │  │
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │  Application              │  │
│  │  - Streamlit App          │  │
│  │  - ChromaDB (local)       │  │
│  │  - Embeddings (local)     │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

### Streamlit Cloud Deployment

```
┌─────────────────────────────────┐
│   Streamlit Cloud               │
│  ┌───────────────────────────┐  │
│  │  Application Container    │  │
│  │  - Streamlit Runtime      │  │
│  │  - Python Dependencies    │  │
│  │  - Secrets Manager        │  │
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │  Data Storage             │  │
│  │  - ChromaDB (from repo)   │  │
│  │  - Embeddings (local)     │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   External Services             │
│  - Groq API (LLM)               │
└─────────────────────────────────┘
```

## Dependencies

### Core Dependencies

- **langchain**: LLM framework
- **langchain-community**: Community integrations
- **langchain-chroma**: ChromaDB integration
- **langchain-text-splitters**: Text chunking
- **langchain-groq**: Groq LLM integration
- **langgraph**: Agent framework

### Data Processing

- **chromadb**: Vector database
- **pypdf**: PDF text extraction
- **sentence-transformers**: Local embeddings

### Web Interface

- **streamlit**: Web framework

### Development

- **python-dotenv**: Environment variable management
- **jupyter**: Notebook development

## Future Architecture Evolution

### Phase 1: Enhanced Features
- Multi-language support
- Real-time document updates
- Advanced analytics dashboard
- Citation formatting improvements

### Phase 2: Scalability
- Multi-user support
- Cloud vector database
- Query caching layer
- Horizontal scaling

### Phase 3: Advanced Capabilities
- Cross-regulation comparison
- Legal citation formatting
- Document versioning
- Advanced search filters

## Conclusion

This architecture provides a solid foundation for a RAG-based AI assistant with excellent performance, zero cost, and high reliability. The use of local processing and free services eliminates cost barriers while maintaining quality. The modular design allows for future enhancements and scaling as needed.
