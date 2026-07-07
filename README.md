# 🇪🇺 AI Ethics Assistant — Expert Agent with Gemini, RAG and LangGraph

A conversational expert agent on **European AI ethics and regulation**, built for the
Generative AI course final project.

## Why this domain?

AI adoption is accelerating faster than public understanding of its risks and rules.
This assistant makes ~250 pages of EU regulation, ethics guidelines and academic
research accessible through natural conversation — for developers, students and
policymakers who need answers, not homework.

## Knowledge Base (4 sources, ~250 pages)

| Document | Type | Covers |
| --- | --- | --- |
| [EU AI Act](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32024R1689) | Legal text | Risk categories, obligations, prohibited practices |
| [A Survey on Bias and Fairness in Machine Learning](https://arxiv.org/abs/1908.09635) (Mehrabi et al.) | Academic paper | Bias types, fairness definitions, mitigation |
| [EP Study: The Ethics of Artificial Intelligence — Issues and Initiatives](https://www.europarl.europa.eu/RegData/etudes/STUD/2020/634452/EPRS_STU(2020)634452_EN.pdf) | Policy study | Real-world case studies (health, justice, warfare) |
| [Ethics Guidelines for Trustworthy AI](https://digital-strategy.ec.europa.eu/en/library/ethics-guidelines-trustworthy-ai) (AI HLEG) | EU framework | 7 requirements for trustworthy AI |

All 4 PDFs ship in `data/` so the notebook runs without any manual download.

## Architecture

PDFs → cleaning → chunking (1400 chars / 150 overlap) → Gemini embeddings
(`gemini-embedding-001`) → **ChromaDB** → **LangGraph** agent
(retrieve → generate) with `gemini-2.5-flash` + **MemorySaver** conversation memory.

## Installation

1. Clone the repo and `cd` into it
2. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. Create a `.env` file in the project root with `GOOGLE_API_KEY=your_key` (get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey))
5. Open `notebooks/ai_ethics_assistant.ipynb` and run top to bottom — the 4 PDFs are already in `data/`

> **Note on quotas:** the indexing cell embeds ~750 chunks and paces itself under Gemini's free-tier limit of 100 requests/minute, but the free tier also caps embeddings at 1000 requests/day. A single fresh run stays under that, but re-running the indexing cell repeatedly (e.g. while experimenting) can exhaust the daily quota — if that happens, wait ~24h or use a billed API key.

### Running the Streamlit app (bonus)

```bash
streamlit run app/streamlit_app.py
```

Requires the vector store to already exist (run the notebook once first to populate `chroma_db/`).

## System Prompt Justification

The system prompt (notebook, Section on agent construction) enforces:

- **Grounding rule:** forces RAG-only answers and an explicit "I don't know" path — prevents hallucinated article numbers, the most dangerous failure mode for a legal/ethics domain.
- **Source attribution:** every answer names which of the 4 documents it came from, so users can verify claims and it's visible that retrieval (not memorized knowledge) is doing the work.
- **Audience adaptation:** answers assume no legal background, since the whole point of the assistant is making ~250 pages of regulation accessible to developers, students and policymakers alike.
- **Brevity:** capped at 2–4 short paragraphs — a chatbot that answers with walls of text defeats its own purpose.
- **Legal disclaimer:** the assistant explicitly says it isn't a lawyer and defers real legal decisions to a professional — responsible-AI practice that's especially fitting for a project *about* AI ethics.

## Requirements

Python 3.9+, Gemini API key. Full dependency list in `requirements.txt`.
