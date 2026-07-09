# AI Ethics Assistant — Presentation Script

**One file to read from while recording.** Combines the visual slide deck
(`presentation.html`, 6 slides — open this in a browser alongside this
script) with the full speaking notes. Read top to bottom; each section names
the slide on screen, then gives the words to say. The live demo and Q&A
prep are at the end. Full technical detail behind every claim here lives in
[README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md), and
[DECISIONS.md](DECISIONS.md) — this script is the spoken layer on top.

**Total run time target: 8–10 minutes** (slides ~4–5 min, live demo ~3 min, Q&A buffer).

---

## Before you hit record

1. Open `presentation.html` in a browser, full-screen, on slide I.
2. Open the deployed app (or `streamlit run app/streamlit_app.py` locally) in a second tab/window, ready but not yet loaded.
3. Have this script open on a second screen or printed — don't read it on the same screen as the slides.

---

## SLIDE I — Preamble

*(On screen: "⚖️ AI Ethics Assistant" — the title slide)*

> "The EU AI Act and its supporting frameworks run to about 250 pages of
> dense legal text. I built a RAG chatbot that answers questions about
> European AI ethics and regulation, grounded only in four authoritative
> source documents — it cites its source on every answer, and it has a
> supervisor layer that checks its own answers before showing them, because
> for a compliance domain a confident wrong answer is worse than 'I don't
> know.'"

---

## SLIDE II — Subject Matter

*(On screen: the 250-pages headline and the 4-document table)*

- **Why this domain:** public value, rich official documentation, addresses a real gap between how fast AI regulation is moving and how few people have actually read it.
- **Knowledge base:** 4 documents, 253 pages (well above the assignment's minimum of 3 docs / ~20 pages):
  1. **EU AI Act** (50 pages) — the legal text itself
  2. **Bias & Fairness Survey**, Mehrabi et al. (34 pages) — technical/academic
  3. **EP Study: Ethics of AI** (128 pages) — policy, real-world case studies
  4. **Ethics Guidelines for Trustworthy AI**, AI HLEG (41 pages) — the EU's own 7-requirement framework
- Deliberately diverse source *types* (legal + academic + policy), not just volume — so the agent can answer regulatory, technical, and ethical questions from the same knowledge base.

> "I didn't just pick one document type. Legal text, academic research, and
> policy analysis all answer different kinds of questions — a regulation
> question, a 'what is bias' question, and a 'what should trustworthy AI
> look like' question all need a different source, and the agent has to
> pick the right one and say so."

---

## SLIDE III — Mechanism

*(On screen: the 5-stage pipeline — Retrieve → Generate → Verify → Revise → Remember)*

- **Retrieve**: MMR search (`fetch_k=20`, `k=4`) for diverse chunks, topped up with plain top-k similarity (a floor under MMR), a bare-question search alongside any conversational-context search, and a keyword lookup for exact article citations. 641 indexed chunks.
- **Generate**: Groq (`llama-3.1-8b-instant`, fast) drafts an answer strictly from the retrieved context.
- **Verify**: a second, *independent* LLM call — on a **stronger** model (`llama-3.3-70b-versatile`) — the supervisor — checks the draft before it's ever shown to the user.
- **Revise**: only runs if verification fails; regenerates once with the specific problem called out.
- **Memory**: LangGraph's `MemorySaver`, keyed by thread ID, so follow-up questions keep context.

> "This is a real two-agent pattern — generator and critic — not just a
> single LLM call with a good prompt. And the critic isn't the same model
> grading its own homework: it's a separate, stronger model, specifically
> because testing showed the fast model wasn't a reliable judge of its own
> paraphrasing."

---

## SLIDE IV — Provisions & Amendments

*(On screen: the tech-stack table + the Gemini amendment callout)*

| Layer | Choice | One-line why |
|---|---|---|
| LLM (generation) | Groq `llama-3.1-8b-instant` | Free, fast, no rate limits |
| LLM (verification) | Groq `llama-3.3-70b-versatile` | Stronger, independent judge of the fast model's drafts |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` | Local, zero cost, no quota |
| Vector store | ChromaDB | Open-source, persistent, simple |
| Agent framework | LangGraph | Stateful graph + built-in memory |
| Interface | Streamlit | Fast to build, free hosting |

### Say the Gemini pivot clearly — don't wait to be asked

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

This preempts the single biggest rubric risk (two criteria literally name Gemini) by owning it instead of hoping it doesn't come up.

---

## SLIDE V — Safeguards

*(On screen: the 3 safeguards list + the calibration dot-plot showing the typo outlier)*

Three independent, complementary layers — not one:

1. **MMR Retrieval** — diverse context instead of four copies of the same passage.
2. **Deterministic Citation Cross-Check** — zero LLM cost, flags a cited source that was never retrieved.
3. **Faithfulness + Relevance Check** — an independent Groq call (on the stronger model) checks both; an evasive answer can't hide behind being technically grounded.

### The repealed provision — a good story about engineering rigor

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

### Bonus material if there's time or it's asked: the post-deployment hardening story

> "After I deployed this, I didn't stop there — I actually used it like a
> real user would, and found six more retrieval bugs a synthetic test set
> wouldn't have caught. For example: asking about a new case study right
> after a different one got answered using the *previous* topic's content,
> because the follow-up-context logic that helps pronoun references like
> 'give an example of it' was hijacking retrieval for questions that
> actually named their own new topic. Another: a typo — 'Autonomus
> Vechicle' — degraded retrieval scores so badly the right answer
> disappeared entirely. I root-caused each one with direct pipeline tests,
> comparing retrieval scores and results before and after, the same
> evidence-based approach as the confidence-gate story. Full data on all
> six in DECISIONS.md, decisions 38 through 43."

---

## SLIDE VI — Findings & Demonstration

*(On screen: the stat blocks — 43 decisions, 641 chunks, $0 cost — and the closing line)*

> "Every retrieval score and verification verdict is logged. The live app's
> sidebar shows a real-time verified/revised tally for the session — a
> number you can watch change, not a promise."

*(Read the closing line on screen, then transition to the live demo.)*

> "The goal wasn't just a working chatbot — it was one that's honest about
> what it doesn't know, and that I can actually prove is honest, live, in
> front of you. Let me show you."

---

## LIVE DEMO — switch to the app now

Ask these in order — each proves a specific point. Wait for each answer and its status badge (✅ Verified / 🔁 Revised) before moving on.

1. **"What is considered a high-risk AI system under the EU AI Act?"**
   → legal text retrieval, citation, ✅ badge.
2. **"What are the main types of bias in machine learning?"**
   → academic source retrieval, different citation.
3. **"What are the seven requirements for trustworthy AI?"**
   → policy framework retrieval.
4. **"What is algorithmic bias?"** then **"Can you give an example of it?"**
   → **memory**, proven live — "it" only resolves correctly because of `MemorySaver`.
5. **"What is the current stock price of Apple?"**
   → out-of-scope refusal — "I don't have enough information..." — the supervisor working without the removed gate.
6. *(Optional, if time)* — open the sidebar, point at the live reliability stats and the "Start new conversation" button.

> "Notice the badge under each answer — verified or revised. That's not
> decoration, that's the supervisor's real verdict on this specific
> answer, right now, live."

---

## Anticipated Q&A

**"Why didn't you use Gemini like the brief said?"**
→ Rate limits blocked development (cite the numbers: 1,000/day, 100/min). Documented pivot, same architecture either way.

**"How do you know it's not hallucinating right now?"**
→ Point at the verified/revised badge under the last answer, and the sidebar stats. It's not a promise, it's a per-answer check you can see.

**"What happens if the supervisor also gets it wrong?"**
→ Honest answer: the deterministic citation check is a hard floor independent of any LLM's judgment; the supervisor is a second, cheap check, not a guarantee. Future work could add quantitative RAGAS-style scoring for a measured hallucination rate.

**"Why remove the confidence gate instead of just tuning the threshold?"**
→ The data showed no threshold could work — a typo scored worse than genuinely irrelevant questions in the same calibration set. That's a structural limit of short-text embedding similarity, not a tuning problem.

**"Did you actually test this after deploying it, or just in development?"**
→ Yes — and that's where six more bugs surfaced (see the bonus material on Slide V). Each was root-caused with direct before/after pipeline tests, not guessed at, and each is documented in DECISIONS.md with the actual data.

**"What would you add with more time?"**
→ Multi-language support, quantitative RAGAS evaluation harness, a labeled test set for measuring faithfulness/relevance scores over time.

---

## Closing Line

> "The goal wasn't just a working chatbot — it was one that's honest about
> what it doesn't know, and that I can actually prove is honest, live, in
> front of you."

*(Stop recording.)*
