# Presentation Notes — AI Ethics Assistant

Speaking notes for the 5–10 minute oral presentation + live demo. This is a
narrative walkthrough of what's already documented in README.md,
ARCHITECTURE.md, and DECISIONS.md — organized in the order you'd actually
say it, mapped to the grading rubric so nothing gets missed live.

---

## 1. The 30-Second Pitch

> "The EU AI Act and its supporting frameworks run to about 250 pages of
> dense legal text. I built a RAG chatbot that answers questions about
> European AI ethics and regulation, grounded only in four authoritative
> source documents — it cites its source on every answer, and it has a
> supervisor layer that checks its own answers before showing them, because
> for a compliance domain a confident wrong answer is worse than 'I don't
> know.'"

---

## 2. Problem & Domain Choice

- **Why this domain:** public value, rich official documentation, addresses
  a real gap between how fast AI regulation is moving and how few people
  have actually read it.
- **Knowledge base:** 4 documents, 253 pages (well above the assignment's
  minimum of 3 docs / ~20 pages):
  1. **EU AI Act** (50 pages) — the legal text itself
  2. **Bias & Fairness Survey**, Mehrabi et al. (34 pages) — technical/academic
  3. **EP Study: Ethics of AI** (128 pages) — policy, real-world case studies
  4. **Ethics Guidelines for Trustworthy AI**, AI HLEG (41 pages) — the EU's
     own 7-requirement framework
- Deliberately diverse source *types* (legal + academic + policy), not just
  volume — so the agent can answer regulatory, technical, and ethical
  questions from the same knowledge base.

---

## 3. Architecture Walkthrough

Say this while pointing at the diagram in README.md / ARCHITECTURE.md:

```
PDF → clean → chunk (1400 chars / 150 overlap) → embed → ChromaDB
                                                              │
User question → Retrieve (MMR) → Generate → Verify → [Finalize | Revise]
```

- **Retrieve**: MMR search (`fetch_k=20`, `k=4`) — diverse chunks, not
  near-duplicates of the same passage — topped up with plain top-k
  similarity, a bare-question search, and a keyword lookup for exact
  article citations (see §7b — these were added after live testing).
- **Generate**: Groq (`llama-3.1-8b-instant`) drafts an answer strictly from
  the retrieved context.
- **Verify**: a second, independent LLM call on a *stronger* model
  (`llama-3.3-70b-versatile`) — the supervisor — checks the draft before
  it's ever shown to the user.
- **Revise**: only runs if verification fails; regenerates once with the
  specific problem called out.
- **Memory**: LangGraph's `MemorySaver`, keyed by thread ID, so follow-up
  questions keep context.

This is a real two-agent (generator + critic) pattern, not just a single
LLM call with a good prompt.

---

## 4. Tech Stack & Why (tell this proactively — don't wait to be asked)

| Layer | Choice | One-line why |
|---|---|---|
| LLM (generation) | Groq `llama-3.1-8b-instant` | Free, fast, no rate limits |
| LLM (verification) | Groq `llama-3.3-70b-versatile` | Stronger, independent judge of the fast model's drafts |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` | Local, zero cost, no quota |
| Vector store | ChromaDB | Open-source, persistent, simple |
| Agent framework | LangGraph | Stateful graph + built-in memory |
| Interface | Streamlit | Fast to build, free hosting |

### The Gemini pivot — say this clearly, don't bury it

> "The assignment brief specifies Gemini for the LLM and embeddings, and
> that's actually what I started with. Gemini's free tier caps embeddings
> at 1,000 requests/day and 100/minute — for a 641-chunk knowledge base,
> that quota blocked me repeatedly during development. I made a deliberate
> call to migrate to Groq for the LLM and local Sentence Transformers for
> embeddings — both genuinely free with no rate limits. The RAG
> architecture, the LangGraph agent design, the memory, and the system
> prompt are all exactly what I would have built on Gemini; only the
> LLM/embedding provider changed. It's documented in DECISIONS.md with the
> specific rate-limit numbers that forced the call."

This preempts the single biggest rubric risk (two criteria literally name
Gemini) by owning it instead of hoping it doesn't come up.

---

## 5. Prompt Engineering — System Prompt Justification

Six rules, each with a specific reason (this is graded explicitly):

1. **Grounding rule** — answer only from context, explicit "I don't know" —
   prevents hallucinated article numbers, the single most dangerous failure
   for a legal domain.
2. **Source attribution** — every answer names its document — lets a human
   verify the claim, demonstrates retrieval is actually working.
3. **Audience adaptation** — explain terms briefly — the whole point is
   making regulation accessible to non-lawyers.
4. **Brevity** — 2–4 paragraphs — a wall of text defeats the purpose.
5. **Legal disclaimer** — not a substitute for professional advice —
   responsible-AI practice for a compliance-adjacent domain.

---

## 6. Conversation Memory (demo this live)

> "Same thread ID → the agent sees the full message history. New thread →
> fresh conversation."

Live sequence that proves it:
1. *"What is algorithmic bias?"*
2. *"Can you give me an example of it?"* — "it" has no meaning without
   memory; if this answers correctly, memory works.
3. *"Which of those is covered by the EU AI Act?"* — refers to the whole
   prior exchange, two turns back.

---

## 7. Hallucination Defenses & Metrics — the core technical depth section

**The framing:** we can't control what a user asks, but we *can* control
what the agent does and doesn't answer when the knowledge base doesn't
support it. Three independent, complementary layers — not one:

| # | Defense | Type | Catches |
|---|---|---|---|
| 1 | MMR retrieval | Retrieval quality | Near-duplicate chunks starving the LLM of real coverage |
| 2 | Deterministic citation cross-check | Rule-based, zero extra LLM calls | A cited source that was never actually retrieved |
| 3 | Supervisor check: faithfulness *and* relevance | Independent Groq call | Fabricated claims, *and* answers that dodge the question while staying technically grounded |

**A fourth layer worth mentioning even though it's not in the final build —
this is a good story about engineering rigor:**

> "I also tried a fourth layer: a pre-generation confidence gate that
> refused to answer if retrieval's similarity score was too weak — fully
> deterministic, zero LLM calls. Calibration looked clean: in-scope
> questions scored 0.43–0.75, out-of-scope scored 1.02–1.78. But real usage
> found a typo — 'What do you mean by **Bais**?' instead of 'bias' — scored
> **1.54**, worse than genuinely out-of-scope questions in my own
> calibration set. No threshold can separate a typo of a real topic from
> something actually irrelevant using embedding similarity alone. I tested
> whether the supervisor alone (no gate) could still handle out-of-scope
> questions correctly — it could — so I removed the gate. I documented the
> full add-then-revert story in DECISIONS.md rather than quietly deleting
> it, because catching your own bad decision with evidence is as important
> as making a good one."

**Metrics language, if asked "how do you measure hallucination":**
- Retrieval confidence is logged per query (Chroma L2 distance — lower is
  more similar).
- Every verification verdict (`VALID` / `INVALID: <reason>`) is logged.
- The Streamlit sidebar shows a live **verified % / revised count / low-
  confidence tally** for the session — a measured number, not an anecdote.
- This maps to RAGAS-style RAG evaluation practice (faithfulness, context
  relevance) without needing the full RAGAS framework — the supervisor
  prompt directly implements both checks.

---

## 7b. Post-Deployment Hardening — a live-testing story worth telling

**This is a strong section if the rubric rewards process, not just a final
demo.** After deploying, real usage (not synthetic test questions) surfaced
a failure class the three defenses above didn't cover: the supervisor was
correctly checking the draft against whatever context `retrieve` handed it —
but sometimes `retrieve` handed it the *wrong* context, so a well-verified
answer still came back "I don't have enough information," even though the
knowledge base genuinely had the answer.

> "After I deployed this, I didn't just call it done — I actually used it
> like a real user would, asked it questions from the source PDFs, and
> found six distinct retrieval bugs that a synthetic test set wouldn't have
> caught. Each one I root-caused with direct pipeline tests before fixing
> it — comparing retrieval scores and results before and after — the same
> evidence-based approach as the confidence-gate story."

The six, briefly (full data in DECISIONS.md #38–43):

1. **Weak verifier** — the fast 8B model judging its own drafts hallucinated
   a faithfulness failure on text that was verbatim in the context. Fixed
   with a separate, stronger judge model.
2. **Follow-up topic hijacking** — asking about a new case study right
   after a different one skewed retrieval toward the *old* topic. Fixed by
   also searching the bare question and merging results.
3. **MMR dropping the best match** — diversity re-ranking excluded the
   single most relevant chunk in favor of a "diverse" but worse one. Fixed
   with a plain top-k similarity floor under MMR.
4. **Bibliography noise** — ~20% of some source PDFs are citation lists
   that share vocabulary with real questions but have no real answer, and
   MMR kept choosing them. Fixed by excluding References sections at
   ingestion (758 → 641 chunks).
5. **Article-number queries** — "What is Article 10?" embeds too poorly to
   retrieve by similarity alone. Fixed with a deterministic keyword lookup.
6. **Typo sensitivity** — "Autonomus Vechicle" wrecked retrieval scores.
   Fixed with a cheap LLM call that normalizes spelling before embedding.

**Why this is worth mentioning even though none of it is in the original
demo script:** it's evidence the reliability claims in §7 aren't just
theoretical — they were pressure-tested against real questions, and when
real usage broke something, it was diagnosed and fixed with data, not
patched blindly.

---

## 8. Reliability Results (show the sidebar live)

During testing, the supervisor caught and corrected real cases:
- An out-of-scope question the generator tried to answer anyway
- A follow-up that cited a source/example not present in the retrieved
  context
- A "grounded but evasive" answer that dodged a direct question about
  compliance penalties

Point at the **Session Reliability** panel during the demo — it's showing
this in real time, not a claim.

---

## 9. Live Demo Script

Ask these in order — each hits a specific rubric point:

1. *"What is considered a high-risk AI system under the EU AI Act?"*
   → legal text retrieval, citation
2. *"What are the main types of bias in machine learning?"*
   → academic source retrieval, different citation
3. *"What are the seven requirements for trustworthy AI?"*
   → policy framework retrieval
4. *"What is algorithmic bias?"* → *"Can you give an example of it?"*
   → **memory**, proven live
5. *"What is the current stock price of Apple?"*
   → out-of-scope refusal — "I don't have enough information..."
6. *(Optional, if time)* — open the sidebar, point at the reliability
   stats and the "Start new conversation" button.

---

## 10. Rubric Self-Assessment (quick reference, don't read verbatim — use if asked)

| Criterion | Status |
|---|---|
| Base de conocimiento (Chroma + Embeddings) | ✅ 641 chunks, correctly indexed — embeddings are Sentence Transformers, not Gemini (see §4) |
| RAG (recuperación + generación) | ✅ precise retrieval, coherent cited answers — LLM is Groq, not Gemini (see §4) |
| Ingeniería de Prompts | ✅ justified in README/ARCHITECTURE, 6 explicit rules |
| Agente con memoria (LangGraph) | ✅ functional, demonstrated live across 3 turns |
| Calidad de código y documentación | ✅ README, ARCHITECTURE.md, DECISIONS.md (43 logged decisions), REVIEW.md |
| Presentación y demo | This document + live app |
| Bonus: Streamlit | ✅ deployed publicly, polished UI |

---

## 11. Anticipated Q&A

**"Why didn't you use Gemini like the brief said?"**
→ Rate limits blocked development (cite the numbers: 1,000/day, 100/min).
Documented pivot, same architecture either way.

**"How do you know it's not hallucinating right now?"**
→ Point at the verified/revised badge under the last answer, and the
sidebar stats. It's not a promise, it's a per-answer check you can see.

**"What happens if the supervisor also gets it wrong?"**
→ Honest answer: the deterministic citation check is a hard floor
independent of any LLM's judgment; the supervisor is a second, cheap check,
not a guarantee. Future work (see Roadmap in README) could add quantitative
RAGAS-style scoring for a measured hallucination rate.

**"Why remove the confidence gate instead of just tuning the threshold?"**
→ The data showed no threshold could work — a typo scored worse than
genuinely irrelevant questions in the same calibration set. That's a
structural limit of short-text embedding similarity, not a tuning problem.

**"What would you add with more time?"**
→ Multi-language support, quantitative RAGAS evaluation harness, a
labeled test set for measuring faithfulness/relevance scores over time.

---

## 12. Closing Line

> "The goal wasn't just a working chatbot — it was one that's honest about
> what it doesn't know, and that I can actually prove is honest, live, in
> front of you."
