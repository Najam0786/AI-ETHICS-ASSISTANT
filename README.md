# AI Ethics Assistant

A conversational AI agent specializing in European AI ethics and regulation, built with Retrieval-Augmented Generation (RAG) and LangGraph.

## Project Overview

This project addresses the critical need for accessible, accurate information about European AI regulation and ethics. As AI adoption accelerates faster than public understanding of its risks and rules, there's a growing gap between technological capability and regulatory knowledge. This assistant bridges that gap by making ~250 pages of EU regulation, ethics guidelines, and academic research accessible through natural conversation.

**Target Audience:** Developers, students, policymakers, and anyone interested in understanding European AI ethics and regulation without needing legal expertise.

## What We Built

A RAG-based conversational agent that:
- Provides **grounded answers** from authoritative sources (no hallucinations)
- **Cites sources** for every response (verifiable information)
- Maintains **conversation context** across multiple questions
- Runs **completely free** using open-source and free-tier services
- Has **no rate limits** or API quotas

## Why We Built This

### Problem Statement
1. **Information Overload**: EU AI Act alone is ~250 pages of complex legal text
2. **Accessibility Gap**: Legal documents are written for lawyers, not developers
3. **Fragmented Sources**: Information scattered across multiple documents (legal, academic, policy)
4. **Hallucination Risk**: General-purpose LLMs can invent legal provisions
5. **Cost Barriers**: Commercial AI APIs have usage limits and costs

### Our Solution
- **RAG Architecture**: Ensures answers are grounded in provided documents
- **Multi-Source Knowledge Base**: Combines legal text, academic research, and policy studies
- **Free Stack**: Eliminates cost barriers using Groq (LLM) and local embeddings
- **Source Attribution**: Every answer cites its source document
- **Conversation Memory**: Maintains context for follow-up questions

## Technology Stack & Rationale

### 1. Groq (LLM)
**What it is:** Fast inference platform providing access to open-source LLMs like Llama 3.1

**Why we chose it:**
- **Completely free**: No usage limits or quotas
- **Fast inference**: Sub-second response times
- **High quality**: Llama 3.1-8b-instant provides excellent performance
- **No rate limiting**: Unlike Gemini or OpenAI free tiers
- **Educational friendly**: Perfect for learning and demonstration projects

**Alternatives considered:**
- Gemini API: Had rate limits (1000 requests/day) that blocked our use case
- OpenAI API: Requires payment, has rate limits
- Claude API: Expensive, not suitable for educational projects

### 2. Sentence Transformers (Embeddings)
**What it is:** Framework for computing sentence embeddings using models like all-MiniLM-L6-v2

**Why we chose it:**
- **Zero cost**: Runs locally, no API calls needed
- **No rate limits**: No daily quotas or request limits
- **High quality**: all-MiniLM-L6-v2 provides excellent semantic understanding
- **Privacy**: No data sent to external services
- **Fast**: Processes 758 chunks in ~13 seconds

**Alternatives considered:**
- OpenAI Embeddings: Costs money, has rate limits
- Gemini Embeddings: Had strict daily quotas (1000 requests/day)
- Cohere Embeddings: Not free for production use

### 3. ChromaDB (Vector Store)
**What it is:** Open-source vector database for storing and retrieving embeddings

**Why we chose it:**
- **Open source**: Completely free, no licensing costs
- **Local deployment**: Runs on your machine, no cloud dependencies
- **Easy integration**: Excellent LangChain support
- **Efficient**: Fast similarity search with minimal overhead
- **Persistent**: Stores data locally, no need for cloud services

**Alternatives considered:**
- Pinecone: Requires cloud account, has usage limits
- Weaviate: More complex setup, cloud-focused
- FAISS: Less user-friendly, no built-in persistence

### 4. LangGraph (Agent Framework)
**What it is:** Framework for building stateful, multi-actor applications with LLMs

**Why we chose it:**
- **Modern architecture**: State-of-the-art agent framework
- **Built-in memory**: MemorySaver for conversation context
- **Flexible**: Easy to define custom agent workflows
- **LangChain integration**: Seamless integration with LangChain components
- **Type safety**: Strong typing with TypedDict for agent state

**Alternatives considered:**
- LangChain Agents: Older, less flexible architecture
- Custom implementation: More complex, less maintainable

### 5. Streamlit (Web Interface)
**What it is**: Python framework for building web applications and data apps

**Why we chose it:**
- **Python-native**: No need for frontend development skills
- **Fast development**: Build UIs in minutes, not hours
- **Built-in components**: Chat interface, file upload, etc.
- **Easy deployment**: One-click deployment to Streamlit Cloud
- **Free hosting**: Streamlit Cloud provides free hosting for small apps

**Alternatives considered:**
- Flask/Django: Requires more development time
- React: Requires frontend development skills
- Gradio: Less flexible for custom UIs

## Knowledge Base

Our knowledge base consists of four authoritative sources covering different aspects of AI ethics:

| Document | Type | Pages | Coverage |
| --- | --- | --- | --- |
| EU AI Act | Legal text | 50 | Risk categories, obligations, prohibited practices |
| Bias & Fairness Survey (Mehrabi et al.) | Academic paper | 34 | Bias types, fairness definitions, mitigation |
| EP Study: Ethics of AI | Policy study | 128 | Real-world case studies (health, justice, warfare) |
| Ethics Guidelines for Trustworthy AI | EU framework | 41 | 7 requirements for trustworthy AI |

**Total:** 253 pages of authoritative content

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   PDF Docs  │───▶│ Text Cleaning│───▶│  Chunking   │───▶│  Embeddings  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                  │
                                                                  ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Response  │◀───│ LangGraph    │◀───│ ChromaDB    │◀───│  Vectors     │
└─────────────┘    │   Agent      │    │ Vector Store│    └──────────────┘
                   └──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │  Groq LLM    │
                   └──────────────┘
```

### Pipeline Details

1. **Document Ingestion**: PDFs loaded using PyPDFLoader
2. **Text Cleaning**: Normalization of whitespace and hyphenation
3. **Chunking**: 1400 characters with 150 overlap (optimized for legal text)
4. **Embedding**: Sentence Transformers (all-MiniLM-L6-v2)
5. **Vector Storage**: ChromaDB with 758 chunks
6. **Retrieval**: Top-K=4 chunks per query
7. **Generation**: Groq LLM with system prompt for grounded responses
8. **Memory**: LangGraph MemorySaver for conversation context

### Key Configuration

- **Chunk Size**: 1400 characters (optimized for legal provisions)
- **Chunk Overlap**: 150 characters (maintains context between chunks)
- **Top-K Retrieval**: 4 chunks (balances context and noise)
- **Temperature**: 0.2 (low creativity for factual accuracy)
- **Embedding Model**: all-MiniLM-L6-v2 (fast, high quality)
- **LLM Model**: llama-3.1-8b-instant (fast, free, high quality)

## Installation

### Prerequisites
- Python 3.9+
- Groq API key (free): [console.groq.com/keys](https://console.groq.com/keys)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Najam0786/AI-ETHICS-ASSISTANT.git
   cd AI-ETHICS-ASSISTANT
   ```

2. **Create virtual environment**
   ```bash
   # Windows
   python -m venv ai_ethics
   ai_ethics\Scripts\activate
   
   # Mac/Linux
   python -m venv ai_ethics
   source ai_ethics/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Build the vector database**
   ```bash
   python build_vector_store.py
   ```
   This processes the PDFs and creates the ChromaDB vector store (~13 seconds).

6. **Run the Streamlit app**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Usage

### Interactive Mode (Streamlit)
Launch the web interface and ask questions about:
- EU AI Act regulations and requirements
- AI bias and fairness in machine learning
- Trustworthy AI requirements and principles
- Ethical considerations in healthcare, justice, warfare, etc.

### Notebook Mode
Open `notebooks/ai_ethics_assistant.ipynb` to:
- Explore the RAG pipeline in detail
- Test retrieval quality with different queries
- Run example queries and analyze results
- Build custom agents or modify the system

## Project Structure

```
ai-ethics-assistant/
├── app/
│   └── streamlit_app.py          # Streamlit web interface
├── data/
│   ├── EU-AI-Act.pdf             # EU AI Act legal text
│   ├── bias_fairness_survey.pdf  # Academic paper on bias
│   ├── ethics_of_ai_study.pdf    # EP study on AI ethics
│   └── trustworthy_ai_guidelines.pdf  # EU ethics guidelines
├── notebooks/
│   └── ai_ethics_assistant.ipynb  # Jupyter notebook with pipeline
├── chroma_db/                      # Vector database (auto-generated)
├── build_vector_store.py          # Script to create vector database
├── requirements.txt              # Python dependencies
├── DECISIONS.md                  # Technical decisions log
├── ARCHITECTURE.md               # Detailed architecture documentation
└── README.md                     # This file
```

## System Prompt Design

The agent follows strict guidelines to ensure reliability and responsible AI practices:

- **Grounding Rule**: Answers only from provided context, explicit "I don't know" for out-of-scope queries
- **Source Attribution**: Every response cites the source document for verification
- **Audience Adaptation**: Explanations designed for non-technical users
- **Brevity**: Concise responses (2-4 paragraphs unless more detail requested)
- **Legal Disclaimer**: Explicitly states not legal advice, recommends professional consultation

This design prevents hallucinated legal claims—the most dangerous failure mode for a legal/ethics domain.

## Performance Metrics

- **Vector Store Creation**: ~13 seconds for 758 chunks (253 pages)
- **Query Response Time**: <2 seconds with Groq LLM
- **Memory Usage**: ~500MB for embeddings model
- **Storage Size**: ~100MB for ChromaDB
- **Retrieval Accuracy**: High (optimized chunking for legal text)
- **Cost**: $0 (completely free stack)

## Deployment

### Local Deployment
Follow the installation steps above to run locally.

### Streamlit Cloud Deployment
The app is deployed at: https://ai-ethics-assistantgit-7qvdmuzrt9xwhgfzbsoub5.streamlit.app/

**Deployment requirements:**
- Groq API key configured in Streamlit Cloud secrets
- ChromaDB vector store included in repository
- All dependencies specified in requirements.txt

## Future Enhancements

Potential improvements for future versions:
- **Multi-language support**: Add support for other EU languages
- **Real-time updates**: Automatically fetch latest EU AI Act updates
- **Advanced analytics**: Track query patterns and popular topics
- **Citation formatting**: Provide proper legal citation format
- **Document expansion**: Add more authoritative sources
- **Comparison features**: Compare different AI regulations globally

## Learning Resources

This project demonstrates several important AI/ML concepts:
- **RAG (Retrieval-Augmented Generation)**: Combining retrieval with generation
- **Vector Databases**: Semantic search with embeddings
- **Agent Frameworks**: Building stateful AI applications
- **Prompt Engineering**: Designing effective system prompts
- **Free AI Stack**: Building production apps without API costs

## License

This project is for educational purposes.

## Author

Nazmul Farooquee

## Acknowledgments

- **EU AI Act**: Official legal text from the European Union
- **Mehrabi et al.**: "A Survey on Bias and Fairness in Machine Learning"
- **European Parliament**: "The Ethics of Artificial Intelligence: Issues and Initiatives"
- **AI HLEG**: "Ethics Guidelines for Trustworthy AI"
- **Groq**: Free LLM inference platform
- **LangChain/LangGraph**: Frameworks for building AI applications
