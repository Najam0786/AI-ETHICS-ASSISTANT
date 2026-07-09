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
| 8 | Sentence Transformers (local) | OpenAI embeddings, Gemini embeddings | Zero cost, no API calls, no rate limits; runs locally; all-MiniLM-L6-v2 provides excellent semantic understanding; processes 641 chunks in ~13 seconds (originally 758, before decision #41 excluded References/Bibliography sections) |

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

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 34 | Custom theme + CSS injection (`.streamlit/config.toml` + `st.markdown` styles) instead of Streamlit defaults | Default Streamlit theme; a full custom component/iframe; switching frameworks | Streamlit's default look is generic and reads as an internal tool rather than a polished product. A custom theme plus targeted CSS (gradient hero title, status pills, styled sidebar cards) gets most of a custom frontend's visual polish for a few dozen lines of CSS, no build step, and no framework migration — consistent with the project's "fast, simple, free" constraints. Verified by actually launching the app and clicking through it in a browser, not just reading the code. |
| 35 | ~~Suggested-question chips~~ — **implemented, then removed** | A separate/duplicated code path for button-triggered questions; keeping the chips permanently visible | First version showed 4 clickable suggested questions before the first message, sharing the same `ask_agent()` path as manual chat input so stats/badges/history stayed consistent. Removed after user feedback: once a question was asked the chips disappeared (by design, to avoid clutter) with no way back to them, which read as a dead end rather than a feature. Simpler and less confusing to have one clear entry point (the chat input) plus an explicit **Start new conversation** button, rather than a UI element that vanishes without explanation. |
| 36 | EU flag colors (Pantone Reflex Blue `#003399` + gold `#FFCC00`) as the theme palette, instead of a generic purple/indigo gradient | Generic indigo/violet gradient (first version); a neutral gray/black theme | User feedback: the initial purple theme looked like "random colors" with no connection to the subject matter. The EU AI Act and the assistant's whole knowledge base are European; using the EU's own flag colors as the palette (blue primary, gold accent, blue→gold gradient divider under the hero) makes the interface visually reinforce what the app is about, not just "attractive in the abstract." |
| 37 | Sidebar reliability stats as custom HTML cards (icon + big number + label, colored left border) plus a gradient progress bar, instead of `st.metric` | `st.metric` (first version) | `st.metric` renders as plain stacked boxes with no visual hierarchy — user feedback called this "really really bad." Custom HTML gives a 2×2 card grid with per-stat accent colors (blue/green/amber/gray) and a blue→gold bar showing verified-percentage at a glance, while still reading the same `st.session_state.stats` dict — no change to the underlying stats logic, only presentation. |

## System Design

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 12 | "I don't know" rule in system prompt | Let model answer freely | Prevents hallucinated legal claims — core responsible-AI choice; ensures accuracy in legal/ethics domain |
| 13 | Source attribution in responses | No source attribution | Enables verification; shows retrieval is working; builds trust; educational value |
| 14 | Conversation memory with MemorySaver | Stateless responses | Maintains context across questions; enables follow-up queries; better user experience |
| 15 | Legal disclaimer in system prompt | No disclaimer | Responsible AI practice; especially important for legal domain; protects users from relying on AI for legal decisions |
| 28 | Supervisor/verification node (generate → verify → revise) | Single-pass generation only; self-critique in one prompt; full multi-agent debate | Testing showed the single-pass agent occasionally fabricated a plausible citation (e.g. an EU AI Act article number) not present in the retrieved context, especially on vague multi-turn follow-ups. A second, independent LLM call re-checks the draft against the same context and forces one regeneration if it finds an unsupported claim. Cheaper and simpler than a full multi-agent debate — one extra Groq call in the common case, two if a correction is needed — while still being a real generator+critic pattern rather than the same model grading its own homework in one shot. Especially important given this is a compliance/ethics assistant, where a confident but ungrounded answer is worse than "I don't know." |

## Hallucination Defenses

The supervisor node (#28) is one layer, not the whole defense. Research into RAG evaluation practice (RAGAS-style faithfulness/relevancy metrics, retrieval-confidence abstention) suggested a **pre-generation confidence gate** would be the single most valuable addition. It was implemented, tested, and then reverted after real usage exposed a flaw the initial calibration missed — documented honestly below as decision #29, since removing a bad decision with evidence is as important to record as adding a good one.

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 29 | ~~Retrieval confidence gate~~ (`retrieve` → `abstain`, threshold 0.9 on Chroma L2 distance) — **implemented, then reverted** | No gate (rely on system prompt's "I don't know" rule only); a fixed low `k` instead of a score check | **Why it was added:** deterministic and free — zero extra LLM calls to reject clearly out-of-scope questions before generation. Initial calibration looked clean: 4 in-scope test questions scored 0.43–0.75, 4 clearly out-of-scope questions (capital of France, cake recipe, stock price, Japanese law) scored 1.02–1.78, and 6 borderline questions (GDPR, neural networks, US AI regulation, penalties for non-compliance) confirmed 0.9 sat cleanly in the gap. **Why it was removed:** real usage surfaced a case the calibration set didn't cover — the typo "What do you mean by **Bais**?" (for "bias") scored **1.54**, worse than "stock price of Apple" (1.42), one of the calibration set's *out-of-scope* examples. No threshold can separate "typo of a real in-scope topic" from "genuinely irrelevant" using embedding similarity alone — that's a property of short-text embeddings, not a tuning mistake. Testing then confirmed `generate → verify` alone already converges correctly to "I don't have enough information" for genuinely out-of-scope questions (stock price, cake recipe), with no gate needed. So the gate was pure downside: it could wrongly block a real, answerable question (a typo any human would read past), while providing no safety benefit the supervisor didn't already provide. `top_score` is still computed and logged for observability, just no longer gates generation. |
| 30 | MMR retrieval (`fetch_k=20`, `lambda_mult=0.5`) instead of plain top-k similarity | Plain top-k similarity (original); re-ranking with a cross-encoder | Plain similarity search can return several near-duplicate chunks from the same passage, effectively giving the LLM less real coverage than TOP_K=4 implies (this is what caused the earlier chroma_db duplication bug to go unnoticed for a while — duplicate chunks looked like normal retrieval results). MMR trades a small amount of pure relevance for diversity, which is a cheap, one-line, no-extra-latency fix; a cross-encoder re-ranker would need a separate model download and inference cost this project's "zero cost, fully local" constraint doesn't want. Also incidentally why the "Bais" typo could still surface the correct Bias & Fairness Survey document in some runs even without the gate — MMR's wider `fetch_k=20` net sometimes pulls it in even when the single top-1 match doesn't. |
| 31 | Deterministic citation cross-check in `verify`, before the LLM call | LLM-only verification (#28) | An LLM verifying another LLM's output is useful but not infallible — testing found cases where the generator cited a source (e.g. "the EU AI Act") that wasn't actually among the retrieved chunks, and a naive verifier didn't reliably catch it either since both are drawing on similar training-data patterns. A plain string check against the known source names retrieved for that query is a hard, non-negotiable floor under the LLM-based check, and costs nothing when it passes (no LLM call needed to confirm a match). |
| 32 | Faithfulness *and* relevance in one verify prompt | Faithfulness-only check (the original #28 prompt) | RAG evaluation practice explicitly warns faithfulness alone is an unreliable signal — a "grounded but evasive" answer can pass a faithfulness-only check while dodging the actual question. Testing confirmed this: a question about penalties for non-compliance got a technically-grounded answer that only suggested other things to ask instead of answering. Extending the same verify call to check both criteria caught this at no extra cost (same single LLM call). |
| 33 | Query-context prepending for retrieval kept despite gate removal | Reverting to raw current-turn text only | The gate's removal didn't undo the fix for conversational follow-ups: a bare question like "give an example of it" still has no self-contained meaning and would retrieve poorly on its own. `retrieve` still prepends the prior assistant turn to the *retrieval query* (not to what `generate` sees) — this improves retrieval quality for follow-ups regardless of whether a gate is present, so it was kept even after the gate itself was reverted. |

**Measurability:** every retrieval score and verification verdict is logged via Python's `logging` module, and the Streamlit UI shows a live per-session count of verified/revised queries plus a low-confidence tally (informational only). This turns "the system seems reliable" into a number that can be reported (e.g., in the live presentation) rather than an anecdotal impression.

| # | Decision | Alternatives Considered | Rationale |
|---|---|---|---|
| 38 | Separate, stronger model (`llama-3.3-70b-versatile`) for the `verify` node, instead of reusing the generation model (`llama-3.1-8b-instant`) | Keep one model for everything; loosen the verify prompt's wording instead | Live testing (screenshots of real chat sessions) showed clearly in-scope questions — "case study about healthcare robots", "what is bias in data?" — getting wrongly answered "I don't have enough information", even though the retrieved context plainly contained the answer. Tracing `retrieve → generate → verify` directly showed retrieval was fine and the draft was well-grounded, but the 8B model acting as its own judge in `verify` hallucinated a faithfulness failure (e.g. flagged "bias in data collection and training" as an invented claim when that exact phrase was in the supplied context). A small, fast model is good at generating fluent drafts but unreliable as a critic of its own paraphrasing. Swapping only the judge to a larger model (generation stays on the fast 8B model for cost/speed) fixed the false positives in testing without loosening the prompt's strictness — which would have reduced its ability to catch real hallucinations (it still correctly flagged a genuine invented "reference to the EU" claim in another test run). |
| 39 | Deterministic keyword fallback (`article_keyword_lookup`, Chroma `where_document={"$contains": ...}`) for bare "Article N" questions, merged into the MMR results | Rely on embedding similarity alone; rephrase the system prompt to ask for clarification | "What is Article 10?" was failing for a different reason than #38: `data/EU-AI-Act.pdf` is a third-party analysis whitepaper ("Trustworthy AI for the Digital Decade", AI & Partners) that only mentions each article as a one-line aside inside paragraphs about unrelated Digital-Decade themes, not as its own section. Testing confirmed none of the top-30 similarity results for "what is Article 10?" contained the literal text — short numeric-citation queries embed too poorly with `all-MiniLM-L6-v2` to surface it via vector search, the same short-text embedding limitation behind decision #29's "Bais" typo finding. Since the exact phrase "Article 10:" is known and cheap to substring-match, a regex on the question (`\bArticle\s+(\d+)\b`) triggers a direct Chroma text-contains lookup, merged into the MMR context before generation — a targeted fix for a specific, confirmed retrieval gap rather than a general-purpose hybrid search layer. |
| 40 | Also retrieve on the bare current-turn question and merge its results into the context, alongside the prior-turn-prepended query (#33) | Only ever search on the prepended query; classify "is this a self-contained follow-up?" before deciding whether to prepend | A follow-up that names its own fully-specified new topic — "give me more details about the case study about Warfare and weaponisation", asked right after an answer about a *different* case study — got its retrieval hijacked by #33's prepending: the combined embedding skewed toward the previous topic and every chunk about warfare vanished from the results entirely (confirmed: a bare search for the same question surfaced the correct chunks; the prepended search returned zero of them). Tried comparing the bare vs. prepended query's top-1 similarity score to decide which to use, but that doesn't work — a longer, prepended query is structurally almost always scored as "closer" to *some* chunk regardless of whether that chunk is actually relevant to the new question, so score comparison can't distinguish "this follow-up needs context" from "this follow-up doesn't." Merging both searches' results sidesteps needing that classification: a self-contained follow-up gets its own topic's chunks back, and a genuine pronoun-style follow-up ("give an example of it") loses nothing since its own bare-question search just contributes little of value alongside the prepended one. |
| 41 | Exclude each source PDF's References/Bibliography section at ingestion (`truncate_at_references` in `build_vector_store.py`, cutting at the first standalone "References"/"Bibliography" heading) | Keep full documents; fix via retrieval-time filtering instead of ingestion; increase `MMR_LAMBDA` toward pure relevance | Investigating the #40 failure further found three of the four source PDFs devote up to ~20% of their pages to citation lists (author names, paper titles, years) that share surface vocabulary with real questions — e.g. bibliography entries mentioning "autonomous vehicles" or "AI" — without containing any substantive answer. Testing showed MMR repeatedly choosing a bibliography chunk as a "diverse" alternative over the actual, more relevant case-study content, even at `lambda_mult=0.7` (heavily relevance-weighted). Cutting the noise at the source, once, when building the vector store is simpler and cheaper than trying to filter it out or out-tune it at every query; chunk count dropped from 758 to 641 chunks after truncation. |
| 42 | Merge in plain top-`TOP_K` similarity search results (in addition to MMR) as a floor under MMR's own picks | Tune `MMR_LAMBDA` further toward relevance; drop MMR entirely | Even after #41 removed the bibliography noise, MMR (`lambda_mult=0.5`, and still true at `0.7`) was found to drop the single best-matching chunk for a query in favor of a less relevant "diverse" one — confirmed directly by comparing plain `similarity_search_with_score` (which reliably included the right chunk in its top 4) against `max_marginal_relevance_search` (which reliably didn't, regardless of lambda tried). Rather than keep hunting for a lambda value that works for every query, plain top-k similarity results are now merged into the context unconditionally: this guarantees the raw best matches are never excluded by MMR's diversity trade-off, while MMR's own selections still add supplementary breadth beyond those guaranteed picks. |
| 43 | LLM-based query normalization (fix obvious spelling errors) before embedding, one extra Groq call in `retrieve` | Leave typo'd queries as-is, relying on generate → verify to say "I don't know" (decision #29's original stance); a hardcoded typo-correction dictionary | A real usage report — "could you please also explain the case study on Autonomus Vechicle" — got wrongly answered "I don't have enough information", the same failure mode as #29's "Bais" example but for an *unambiguous* typo. Direct testing confirmed the typo alone wrecks retrieval: `all-MiniLM-L6-v2` scored the typo'd query 1.37-1.55 against the collection (no relevant chunk in the top 6), versus 0.96-1.06 with the correct chunk at rank 2 once corrected. Decision #29 deliberately avoided a retrieval-score gate because scores can't reliably separate "typo of an in-scope topic" from "genuinely out-of-scope" — but that finding is about *gating on the score*, not about *fixing the input before embedding*. A cheap, dedicated LLM call (temperature 0, tightly scoped prompt: fix spelling only, don't answer, don't add information) corrects unambiguous typos like "Vechicle" → "Vehicle" while leaving genuinely ambiguous ones like "Bais" untouched in testing — consistent with #29's finding that ambiguous cases are correctly left for generate → verify to resolve. Only the retrieval query is affected; `generate` still sees the user's original wording via the full message history. |

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
