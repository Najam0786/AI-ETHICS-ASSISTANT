# ⚖️ AI Ethics Assistant

**A conversational AI agent that makes European AI regulation understandable — grounded, cited, and free to run.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://ai-ethics-assistantgit-7qvdmuzrt9xwhgfzbsoub5.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-1C3C3C?style=flat)](https://www.langchain.com/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=flat)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)

**🔗 [Try the live app](https://ai-ethics-assistantgit-7qvdmuzrt9xwhgfzbsoub5.streamlit.app/)** · **📦 [Source on GitHub](https://github.com/Najam0786/AI-ETHICS-ASSISTANT)**

> **⚠️ Note on the LLM/embeddings stack:** this project was originally built on Google Gemini for both the LLM and embeddings. Gemini's free-tier quota (1,000 embedding requests/day, 100 requests/minute) repeatedly blocked development, so the stack was deliberately migrated to **Groq** (LLM) + **local Sentence Transformers** (embeddings) — both genuinely free with no rate limits. The RAG architecture, LangGraph agent, memory, and system prompt design are all unchanged. Full rationale and alternatives considered in [DECISIONS.md](DECISIONS.md#technology-stack).

---

## 📖 Overview

The EU AI Act and its supporting frameworks run to **~250 pages of dense legal and policy text**. Most people who need to understand it — developers, students, policymakers — don't have the time or legal background to read it cover to cover.

**AI Ethics Assistant** is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about European AI ethics and regulation using *only* its authoritative knowledge base — never guessing, always citing its source.

| | |
|---|---|
| ✅ **Grounded answers** | Every response comes from the source documents — no hallucinated legal claims |
| 🛡️ **Supervised** | A second LLM independently checks each draft for faithfulness *and* relevance before it's shown |
| 🔍 **Citation-checked** | A deterministic check catches a cited source that wasn't actually retrieved |
| 📚 **Source citations** | Each answer names the document it's drawn from |
| 💬 **Conversational memory** | Follow-up questions keep context via LangGraph |
| 📊 **Measurable reliability** | Every query logs its retrieval confidence and verification verdict — a real verified/revised rate, not a guess |
| 🆓 **Zero cost to run** | Free-tier LLM (Groq) + local embeddings — no API bills, no rate limits |

---

## 🧠 How It Works

```text
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   PDF Docs  │───▶│ Text Cleaning│───▶│  Chunking   │───▶│  Embeddings  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                  │
                                                                  ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Response  │◀───│   Finalize   │◀───│   Verify    │◀───│   Retrieve   │
└─────────────┘    └──────▲───────┘    │ (citation + │    │ (MMR + Groq) │
                          │            │  faith/rel) │    └──────▲───────┘
                    ┌─────┴─────┐      └──────┬──────┘           │
                    │  Revise   │◀────invalid──┘                 │
                    └───────────┘                          ┌─────┴──────┐
                                                             │  Generate  │
                                                             │   (Groq)   │
                                                             └────────────┘
```

1. **Ingest** — PDFs parsed with PyPDFLoader
2. **Clean & chunk** — normalized text split into 1,400-char chunks (150-char overlap)
3. **Embed** — Sentence Transformers (`all-MiniLM-L6-v2`), fully local
4. **Store** — ChromaDB, persisted to disk (641 chunks; each source's References/Bibliography section is dropped at ingestion — see [Hallucination Defenses](#hallucination-defenses))
5. **Retrieve** — MMR search for 4 diverse relevant chunks, topped up with plain top-k similarity results (a floor under MMR's diversity trade-off), a bare-question search alongside any conversational-context search, and a deterministic keyword lookup for exact article citations — plus a separate top-1 similarity score logged as a confidence signal
6. **Generate** — Groq (`llama-3.1-8b-instant`) drafts an answer strictly from retrieved context
7. **Verify** — a deterministic citation check, then a second, independent Groq call on a stronger model (`llama-3.3-70b-versatile`) checking faithfulness *and* relevance
8. **Revise** *(only if the check fails)* — regenerates the answer, told exactly what was wrong
9. **Remember** — LangGraph `MemorySaver` keeps conversation state across turns

📄 Full technical write-up in [ARCHITECTURE.md](ARCHITECTURE.md) · Design rationale in [DECISIONS.md](DECISIONS.md)

---

## 🛡️ Hallucination Defenses

**The core problem:** we can't control what a user asks, but we *can* control what the agent does and doesn't answer when the knowledge base doesn't genuinely support it. A system prompt telling the model to say "I don't know" is a request, not a guarantee — testing surfaced real cases where the LLM answered anyway, citing a source that was never retrieved, or dodging the actual question while staying technically "grounded." So the agent uses independent, complementary checks instead of trusting one model's judgment:

| # | Defense | Type | Catches |
|---|---|---|---|
| 1 | **MMR retrieval** (vs. plain top-k similarity) | Retrieval quality | Near-duplicate chunks crowding out genuinely diverse context, which pushes the LLM to "fill gaps" with invented detail |
| 2 | **Deterministic citation cross-check** | Rule-based, zero LLM calls in the common case | A cited source that was never actually retrieved — checked by plain string matching before any LLM judges the draft |
| 3 | **Supervisor LLM check (faithfulness *and* relevance)**, on a stronger model (`llama-3.3-70b-versatile`) than generation | Second independent Groq call | Fabricated claims not in context (faithfulness), *and* answers that are technically grounded but dodge the actual question (relevance) — faithfulness alone can pass an evasive non-answer |

**A fourth layer was tried and removed:** a hard pre-generation gate that abstained whenever retrieval confidence exceeded a calibrated threshold, so obviously out-of-scope questions never reached the LLM at all. It worked for clean cases, but real usage exposed a problem: a typo as small as *"Bais"* instead of *"bias"* scored **worse** than several genuinely out-of-scope test questions — no threshold could separate "typo of a real topic" from "actually irrelevant" using embedding similarity alone, so the gate was blocking legitimate questions. Since `generate → verify` was already proven to correctly handle genuinely out-of-scope questions on its own, the gate added real risk (false refusals) for no safety benefit, so it was removed. The retrieval score is still computed and logged for observability, just no longer used to block generation. Full before/after data in [DECISIONS.md](DECISIONS.md#hallucination-defenses).

### Retrieval-quality fixes found through live testing

Live use surfaced several cases where the *answer* was correctly refused as "not enough information" — but the knowledge base actually had the answer, and the failure was upstream in retrieval, not in the verifier. Each was root-caused with direct pipeline testing (not guessed at) and fixed:

| Symptom | Root cause | Fix |
|---|---|---|
| A follow-up naming a brand-new topic (e.g. a different case study) got answered using the *previous* topic's content | Prepending the prior conversational turn to the retrieval query — meant to help pronoun-style follow-ups — hijacked retrieval for follow-ups that actually named their own fully-specified topic | Also retrieve on the bare current-turn question and merge results in |
| A clearly relevant chunk was excluded even though it plainly matched | MMR's diversity trade-off dropped the single best match for a "diverse" but less relevant one, confirmed at multiple `lambda_mult` values | Merge in plain top-k similarity results as a floor under MMR's picks |
| Up to ~20% of some source PDFs (citation lists) were being retrieved over genuinely relevant content | Bibliography entries share surface vocabulary ("autonomous vehicles", "AI") with real questions without containing any real answer | Exclude each document's References/Bibliography section at ingestion |
| "What is Article 10?" failed even though the article is discussed in the source | Short, numeric-citation queries embed poorly with `all-MiniLM-L6-v2` — none of the top-30 similarity results contained the literal text | Deterministic keyword lookup for exact "Article N" citations |
| A typo ("Autonomus Vechicle") returned "not enough information" | The typo alone degraded retrieval scores from ~0.96 to ~1.5, and the correct chunk vanished from the results | One cheap LLM call normalizes obvious spelling errors before embedding |
| The same small model judged its own drafts and hallucinated a "faithfulness" failure on text that was actually verbatim in the context | A fast/small model is a good drafter but an unreliable critic of its own paraphrasing | `verify` now uses a separate, stronger model than `generate` |

Full root-cause data and alternatives considered for each: [DECISIONS.md](DECISIONS.md#hallucination-defenses) (decisions #38–43).

Every retrieval score and verification verdict is logged, and the Streamlit sidebar shows a live **Session Reliability** panel — verified/revised counts and a low-confidence tally — so reliability is a measured number, not an impression.

---

## 📚 Knowledge Base

| Document | Type | Pages | Covers |
|---|---|---|---|
| **EU AI Act** | Legal text | 50 | Risk categories, obligations, prohibited practices |
| **A Survey on Bias and Fairness in ML** (Mehrabi et al.) | Academic paper | 34 | Bias types, fairness definitions, mitigation |
| **The Ethics of AI: Issues and Initiatives** (European Parliament) | Policy study | 128 | Case studies — health, justice, warfare |
| **Ethics Guidelines for Trustworthy AI** (AI HLEG) | EU framework | 41 | 7 requirements for trustworthy AI |

**Total: 253 pages** of authoritative source material (each document's References/Bibliography section is excluded at ingestion — see [Hallucination Defenses](#hallucination-defenses) — leaving 641 indexed chunks).

---

## 🛠️ Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **LLM (generation)** | [Groq](https://groq.com/) — `llama-3.1-8b-instant` | Free, no rate limits, sub-second inference |
| **LLM (verification)** | [Groq](https://groq.com/) — `llama-3.3-70b-versatile` | A stronger model as an independent judge of the fast model's drafts — still free |
| **Embeddings** | [Sentence Transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` | Runs locally, zero cost, no quotas |
| **Vector store** | [ChromaDB](https://www.trychroma.com/) | Open-source, persistent, simple LangChain integration |
| **Agent framework** | [LangGraph](https://www.langchain.com/langgraph) | Stateful graph with built-in conversation memory |
| **Interface** | [Streamlit](https://streamlit.io/) | Fast to build, free cloud hosting |

Full rationale and alternatives considered → [DECISIONS.md](DECISIONS.md)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A free [Groq API key](https://console.groq.com/keys)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Najam0786/AI-ETHICS-ASSISTANT.git
cd AI-ETHICS-ASSISTANT

# 2. Create and activate a virtual environment
python -m venv ai_ethics
# Windows
ai_ethics\Scripts\activate
# Mac/Linux
source ai_ethics/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# 5. Build the vector store (~13 seconds)
python build_vector_store.py

# 6. Launch the app
streamlit run app/streamlit_app.py
```

### 💡 Usage

Open the app and ask things like:

- *"What are the prohibited AI practices under the EU AI Act?"*
- *"How does the Act define high-risk AI systems?"*
- *"What are the 7 requirements for trustworthy AI?"*
- *"What are common sources of bias in machine learning models?"*

Each answer shows a colored status pill (✅ verified / 🔁 revised), and the sidebar's **Session Reliability** panel shows live stat cards plus a verified-percentage bar for the whole session — with a **🔄 Start new conversation** button to reset.

Or explore [`notebooks/ai_ethics_assistant.ipynb`](notebooks/ai_ethics_assistant.ipynb) to inspect the RAG pipeline step by step.

---

## 📁 Project Structure

```
ai-ethics-assistant/
├── app/
│   └── streamlit_app.py          # Streamlit web interface + LangGraph agent
├── .streamlit/
│   └── config.toml               # Custom theme (colors, font)
├── data/                         # Source PDFs (EU AI Act, bias survey, EP study, ethics guidelines)
├── notebooks/
│   └── ai_ethics_assistant.ipynb # RAG pipeline walkthrough
├── chroma_db/                    # Persisted vector store
├── build_vector_store.py         # Ingestion → chunking → embedding → storage script
├── requirements.txt
├── ARCHITECTURE.md               # Detailed technical architecture
├── DECISIONS.md                  # Technology choices & rationale
└── README.md
```

---

## 🛡️ Responsible AI Design

- **Grounding rule** — the agent answers *only* from retrieved context; it explicitly says "I don't have enough information" rather than guessing (see [Hallucination Defenses](#hallucination-defenses) for the four independent layers enforcing this)
- **Source attribution** — every answer names the document it came from
- **Plain-language explanations** — written for non-lawyers
- **Legal disclaimer** — the assistant is not a substitute for professional legal advice

---

## 📊 Performance

| Metric | Value |
|---|---|
| Vector store build time | ~13s for 641 chunks |
| Query response time | < 3s (one extra Groq call for query normalization, plus a stronger-model verify call) |
| Storage footprint | ~100MB |
| Running cost | **$0** |

---

## 🗺️ Roadmap

- [ ] Multi-language support for other EU languages
- [ ] Automatic sync with EU AI Act updates
- [ ] Formal legal citation formatting
- [ ] Query analytics dashboard
- [ ] Cross-jurisdiction regulation comparison

---

## 📜 License

Released under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **European Union** — EU AI Act (official legal text)
- **Mehrabi et al.** — *A Survey on Bias and Fairness in Machine Learning*
- **European Parliament** — *The Ethics of Artificial Intelligence: Issues and Initiatives*
- **AI HLEG** — *Ethics Guidelines for Trustworthy AI*
- **Groq** & **LangChain/LangGraph** — the free, open tooling that made this possible

---

<p align="center">Built by <strong>Nazmul Farooquee</strong></p>
