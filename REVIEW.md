# Code Review — Pre-Submission Audit

A high-effort code review of the working-tree changes to `app/streamlit_app.py` and
`build_vector_store.py` (decisions #38–43 in [DECISIONS.md](DECISIONS.md): the
dual-model verifier, query normalization, retrieval-merge fixes, and the
References/Bibliography exclusion at ingestion). Eight independent finder passes
(3 correctness, 3 cleanup, 1 altitude, 1 conventions) surfaced ~25 raw candidates;
after deduplication and one independent verification pass per candidate, 9 survive
below, ranked most-severe first. None are regressions in the *user-facing* behavior
fixed this session — all are either scope/robustness gaps around the new code or
documentation that fell out of sync.

## Findings

| # | Verdict | File | Finding |
|---|---|---|---|
| 1 | CONFIRMED | `app/streamlit_app.py` (`retrieve`) | `docs` is merged from up to 4 separate retrieval passes (prefixed MMR, bare-question MMR, plain top-k similarity, article-keyword lookup), deduped only by exact `page_content`, and never capped back to `TOP_K`. In the worst case (a topic-switching follow-up that also cites an article number) context can reach ~16 chunks instead of the intended 4, inflating prompt size/cost for `generate` and `verify` well beyond what `TOP_K=4` implies elsewhere in the code and docs. |
| 2 | CONFIRMED | `app/streamlit_app.py` (`retrieve`) | `top_score` (the sidebar's "Low Confidence" stat) is now computed from `normalized_question` (the LLM-corrected spelling) instead of the user's raw input. A typo that would previously have scored badly and been flagged low-confidence (per decision #29's own calibration data) now scores well after correction, silently changing what the metric measures with no comment/doc update acknowledging the shift. |
| 3 | CONFIRMED | `app/streamlit_app.py` (`retrieve`) | Two redundant Chroma round trips: (a) the top-1 confidence lookup and the plain top-k floor query identical text whenever there's no prior AI turn — the k=1 result is a strict prefix of the later k=TOP_K call; (b) whenever there *is* a prior turn, `normalized_question` is independently embedded and queried twice (once via MMR, once via plain similarity) within the same `retrieve()` call. |
| 4 | CONFIRMED | `app/streamlit_app.py` (`retrieve`) | Three near-identical "merge new results into `docs`, dedupe via `seen`" blocks (bare-MMR merge, plain-similarity merge, keyword-lookup merge) are copy-pasted with only the loop variable name differing, and the MMR call's argument list (`k`, `fetch_k`, `lambda_mult`) is duplicated verbatim across two call sites. A future fix to the merge/dedup logic has to be applied in 3+ places consistently. |
| 5 | CONFIRMED — **fixed** | `app/streamlit_app.py` (`main`, "About this assistant" panel) | The sidebar still described a single LLM (`llama-3.1-8b-instant`) even though `load_agent()` instantiates a second, independent judge model (`llama-3.3-70b-versatile`) that verifies every answer. Fixed: the panel now lists both models separately. |
| 6 | PLAUSIBLE | `app/streamlit_app.py` (`retrieve`, `ask_agent`) | `ask_agent()` has no `try/except` around `agent.invoke()`. `retrieve()` now makes an LLM call (query normalization) before any other node runs — a transient Groq failure there now surfaces earlier than before. Note: `verify()` already made an unguarded LLM call pre-diff, so this spreads a pre-existing unhandled-exception risk to one more (earlier) call site rather than introducing a wholly new risk class. |
| 7 | PLAUSIBLE | `build_vector_store.py` (`truncate_at_references`) | Truncates a document at the *first* page matching a bare "References"/"Bibliography" heading line, with no logging of before/after page counts and no safeguard against a false-positive match (e.g. a table header, a per-chapter subsection, a stray ToC line) silently dropping real content. Verified empirically correct for all 4 current source PDFs (checked actual extracted text — each match lands exactly on the true bibliography start), but there's nothing to catch a bad match if a different PDF is added later. |
| 8 | PLAUSIBLE | `app/streamlit_app.py` (`article_keyword_lookup`) | The "Article N:" substring format is confirmed consistent across the current corpus, but `data/EU-AI-Act.pdf` is only a 50-page third-party *summary* (not the full ~150-article legal text), so many article numbers a user might reasonably ask about have no dedicated citation chunk at all. The lookup silently returns `[]` in that case, and neither it nor `retrieve()` logs a hit/miss count, so there's no way to tell from the logs whether this fallback fired. |
| 9 | PLAUSIBLE — **fixed** | `DECISIONS.md` (decision #8) | Still read "processes 758 chunks in ~13 seconds," left unedited even though decision #41 (added this session, further down the same file) documents the count dropping to 641. Fixed: updated to 641 with a note on the original count. |

## What Wasn't Changed

Findings 1–4 and 6 are about the *shape* of `retrieve()` after six fixes accreted onto it this session (documented in DECISIONS.md #38–43, each added to fix one specific, evidence-backed failure). None of them are asked to be fixed as part of this review — they're recorded here as known, scoped trade-offs for anyone picking this codebase up next: the multi-pass merge is deliberate hybrid retrieval (not a bug), but it isn't capped, isn't deduplicated of the redundant round trips, and isn't refactored into a shared helper yet.

Findings 5 and 9 are one-line documentation fixes. Findings 7 and 8 are accepted, evidence-checked scope limits (correct for the current 4 PDFs / current corpus), not live bugs.

One additional candidate — a concern that `article_keyword_lookup`'s use of Chroma's private `_collection` attribute and unpinned `chromadb`/`langchain-chroma` versions in `requirements.txt` could break on every question — was investigated and **refuted**: the regex check happens *before* the risky call, so it only ever executes on questions that actually mention "Article N," not unconditionally. The unpinned-version fact is real but the blast radius is narrower than initially suspected.
