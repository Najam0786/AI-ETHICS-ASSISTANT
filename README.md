# AI Ethics Assistant

A conversational AI agent specializing in European AI ethics and regulation, built with RAG (Retrieval-Augmented Generation) and LangGraph.

## Overview

This assistant provides accurate, grounded answers about European AI regulation, ethics guidelines, and academic research. It leverages a knowledge base of ~250 pages from authoritative sources including the EU AI Act, academic papers, and policy studies.

## Features

- **RAG-based responses**: Grounded in authoritative sources, not hallucinations
- **Source attribution**: Every answer cites its source document
- **Conversation memory**: Maintains context across multiple questions
- **Free to run**: Uses Groq (LLM) and local Sentence Transformers (embeddings)
- **No rate limits**: Completely free with no API quotas

## Knowledge Base

| Document | Type | Coverage |
| --- | --- | --- |
| EU AI Act | Legal text | Risk categories, obligations, prohibited practices |
| Bias & Fairness Survey (Mehrabi et al.) | Academic paper | Bias types, fairness definitions, mitigation |
| EP Study: Ethics of AI | Policy study | Real-world case studies (health, justice, warfare) |
| Ethics Guidelines for Trustworthy AI | EU framework | 7 requirements for trustworthy AI |

## Architecture

```
PDFs → Text Cleaning → Chunking → Embeddings → ChromaDB → LangGraph Agent → Response
```

- **Chunking**: 1400 characters with 150 overlap
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2) - local, no API calls
- **Vector Store**: ChromaDB with 758 chunks from 253 pages
- **LLM**: Groq (llama-3.1-8b-instant) - free, no quotas
- **Agent Framework**: LangGraph with MemorySaver for conversation memory

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
- EU AI Act regulations
- AI bias and fairness
- Trustworthy AI requirements
- Ethical considerations in healthcare, justice, etc.

### Notebook Mode
Open `notebooks/ai_ethics_assistant.ipynb` to:
- Explore the RAG pipeline
- Test retrieval quality
- Run example queries
- Build custom agents

## Project Structure

```
ai-ethics-assistant/
├── app/
│   └── streamlit_app.py          # Streamlit web interface
├── data/
│   ├── EU-AI-Act.pdf
│   ├── bias_fairness_survey.pdf
│   ├── ethics_of_ai_study.pdf
│   └── trustworthy_ai_guidelines.pdf
├── notebooks/
│   └── ai_ethics_assistant.ipynb  # Jupyter notebook with pipeline
├── build_vector_store.py         # Script to create vector database
├── requirements.txt              # Python dependencies
├── DECISIONS.md                  # Technical decisions log
└── README.md                     # This file
```

## System Prompt Design

The agent follows strict guidelines to ensure reliability:

- **Grounding**: Answers only from provided context, explicit "I don't know" for out-of-scope queries
- **Source attribution**: Every response cites the source document
- **Audience adaptation**: Explanations for non-technical users
- **Brevity**: Concise responses (2-4 paragraphs unless detailed)
- **Legal disclaimer**: Not legal advice, recommends professional consultation

## Technical Details

### Why This Stack?

- **Groq**: Free, fast LLM with no rate limits
- **Sentence Transformers**: Local embeddings, zero cost, no API calls
- **ChromaDB**: Efficient vector storage, open-source
- **LangGraph**: Modern agent framework with built-in memory management

### Performance

- **Vector store creation**: ~13 seconds for 758 chunks
- **Query response**: <2 seconds with Groq
- **Memory usage**: ~500MB for embeddings model
- **Storage**: ~100MB for ChromaDB

## License

This project is for educational purposes.

## Author

Nazmul Farooquee
