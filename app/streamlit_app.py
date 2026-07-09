"""
AI Ethics Assistant - Streamlit Web Interface

A RAG-based conversational agent for European AI ethics and regulation.
Uses Groq for LLM and local Sentence Transformers for embeddings.
"""

import logging
import os
import re
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("ai_ethics_assistant")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

CHROMA_DIR = str(ROOT_DIR / "chroma_db")
COLLECTION_NAME = "ai_ethics_eu"
LLM_MODEL = "llama-3.1-8b-instant"
# The verify node is a judge task, not a generation task: testing found the 8B model
# used for generation hallucinates faithfulness failures on its own correctly-grounded
# paraphrases (e.g. flagged "bias in data collection and training" as an invented claim
# when that exact phrase was in the context). A larger model as the sole judge fixed it.
VERIFY_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4
FETCH_K = 20            # candidates considered by MMR before diversity re-ranking
MMR_LAMBDA = 0.5        # 0 = max diversity, 1 = pure relevance

# Bare "Article N" queries embed poorly against a knowledge base document that only
# mentions articles in passing one-liners rather than as their own dedicated sections —
# none of the top-30 similarity results contain the literal text for some article
# numbers. A deterministic keyword lookup catches exact article citations that
# embedding similarity alone misses.
ARTICLE_RE = re.compile(r"\barticle\s+(\d+)\b", re.IGNORECASE)

# Chroma's default distance is squared L2 (lower = more similar). A hard
# pre-generation abstain gate on this score was tried and removed: typo'd or
# very short in-scope questions (e.g. "What do you mean by Bais?", a typo for
# "bias") scored *worse* (1.5+) than several genuinely out-of-scope questions,
# so no threshold could separate the two without also blocking legitimate
# questions. generate -> verify already handles out-of-scope questions
# correctly (proven in testing), so top_score is kept only as a logged/
# displayed confidence signal, not a gate. See DECISIONS.md for the data.
LOW_CONFIDENCE_LOG_THRESHOLD = 1.0  # informational label only, never blocks generation

SYSTEM_PROMPT = """You are an expert assistant on European AI ethics and regulation.

Your knowledge base consists of four authoritative sources:
1. The EU AI Act (official legal text)
2. "A Survey on Bias and Fairness in Machine Learning" (Mehrabi et al.)
3. The European Parliament study "The Ethics of Artificial Intelligence: Issues and Initiatives"
4. The EU "Ethics Guidelines for Trustworthy AI" (AI HLEG)

RULES:
1. Answer ONLY using the context provided below. Never invent facts, article numbers, or legal requirements.
2. If the context does not contain the answer, say clearly: "I don't have enough information in my knowledge base to answer that" — and suggest what the user could ask instead.
3. Mention which source your answer is based on (e.g., "According to the EU AI Act...").
4. Be clear and educational: your users may be developers, students, or policymakers with no legal background. Explain technical or legal terms briefly when you use them.
5. Be concise: 2–4 short paragraphs maximum, unless the user asks for more detail.
6. You are not a lawyer. For legal decisions, recommend consulting a qualified professional.

CONTEXT FROM KNOWLEDGE BASE:
{context}
"""

VERIFY_PROMPT = """You are a strict compliance reviewer for an AI ethics assistant. Check the \
DRAFT ANSWER against two criteria:

1. Faithfulness: every claim is directly supported by the CONTEXT — no invented facts, article \
numbers, or legal claims that aren't there.
2. Relevance: the draft actually answers the QUESTION asked, rather than dodging it or drifting \
onto a related but different topic.

QUESTION:
{question}

CONTEXT:
{context}

DRAFT ANSWER:
{draft}

Reply with exactly "VALID" if both criteria are met (or the draft correctly states it doesn't \
have enough information to answer). Otherwise reply with "INVALID: " followed by a one-sentence \
explanation, noting whether it's a faithfulness or relevance failure.
"""

NORMALIZE_QUERY_PROMPT = """Fix any spelling or typing errors in the QUESTION below. Do not \
change its meaning, do not answer it, and do not add information. If there are no errors, \
repeat it exactly as-is.

QUESTION: {question}

Reply with ONLY the corrected question, nothing else."""

REVISE_PROMPT = """Your previous draft failed a review: {reason}

Rewrite the answer to the question below using ONLY the context provided, correcting this \
issue. If the context doesn't actually support an answer, say so explicitly.

QUESTION: {question}
"""

# Canonical source names (must match metadata["source_name"] from build_vector_store.py) mapped
# to the distinctive phrases the system prompt encourages the LLM to cite them by. Used as a
# deterministic backstop alongside the LLM-based verify step: if the draft cites a source by name
# but that source isn't among what was actually retrieved, that's a citation hallucination no
# LLM self-check is needed to catch.
SOURCE_CITATION_KEYWORDS = {
    "EU AI Act": ["EU AI Act"],
    "Bias & Fairness Survey (Mehrabi et al.)": ["Mehrabi"],
    "EP Study: Ethics of AI": ["European Parliament study", "Issues and Initiatives"],
    "Ethics Guidelines for Trustworthy AI": [
        "AI HLEG",
        "Ethics Guidelines for Trustworthy AI",
        "Trustworthy AI Guidelines",
    ],
}


def article_keyword_lookup(question: str, vectorstore: Chroma) -> list[Document]:
    """Deterministic substring fallback for exact 'Article N' citations, which short
    embedding queries often fail to retrieve by similarity alone (see ARTICLE_RE)."""
    match = ARTICLE_RE.search(question)
    if not match:
        return []
    result = vectorstore._collection.get(
        where_document={"$contains": f"Article {match.group(1)}:"},
        limit=4,
    )
    return [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(result["documents"], result["metadatas"])
    ]


def check_citations(draft: str, retrieved_sources: list[str]) -> Optional[str]:
    """Deterministic backstop: flag a source the draft cites that wasn't actually retrieved."""
    for source_name, keywords in SOURCE_CITATION_KEYWORDS.items():
        if any(kw in draft for kw in keywords) and source_name not in retrieved_sources:
            return (
                f"cites '{source_name}' but it was not among the retrieved sources "
                f"({', '.join(retrieved_sources) or 'none'})"
            )
    return None


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    draft: str
    verification: str
    top_score: float
    sources_retrieved: list[str]


class LocalEmbeddings:
    """Local sentence-transformers embeddings wrapper for LangChain compatibility."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()


def load_agent():
    """Initialize and return the RAG agent with vector store and LLM."""

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not found — check your .env file")

    if not Path(CHROMA_DIR).exists():
        raise RuntimeError(
            "Vector store not found. Run 'python build_vector_store.py' first."
        )

    # Initialize embeddings and vector store
    embeddings = LocalEmbeddings(EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Initialize LLMs: a fast model for generation, a stronger one to judge it (see
    # VERIFY_MODEL comment above for why these are split).
    llm = ChatGroq(model=LLM_MODEL, temperature=0.2)
    verifier_llm = ChatGroq(model=VERIFY_MODEL, temperature=0.2)

    # Define agent nodes
    def retrieve(state: AgentState) -> dict:
        """Retrieve relevant documents, using MMR for diversity and plain similarity for a
        confidence score (MMR re-ranks for diversity, so its distances aren't a clean
        relevance signal — a separate top-1 lookup gives us that)."""
        question = state["messages"][-1].content

        # all-MiniLM-L6-v2 is fragile on typo'd short queries — "Autonomus Vechicle"
        # scored 1.37-1.55 against the collection (no relevant chunk in the top 6),
        # while the corrected spelling scored 0.96-1.06 with the right chunk at rank
        # 2 (confirmed by direct testing). One cheap Groq call fixes obvious spelling
        # errors before embedding; generate() still sees the user's raw question via
        # the full message history, so this only affects retrieval, not what's shown.
        normalize_check = llm.invoke(
            [HumanMessage(content=NORMALIZE_QUERY_PROMPT.format(question=question))]
        )
        normalized_question = normalize_check.content.strip() or question

        # A bare follow-up like "give an example of it" has no semantic content on its
        # own and always scores poorly, which would wrongly trigger abstention on every
        # pronoun-referencing follow-up. Prepend the prior assistant answer (if any) to
        # give retrieval enough context to resolve it — generate() still sees the raw
        # question via the full message history, so this only affects retrieval.
        retrieval_query = normalized_question
        for msg in reversed(state["messages"][:-1]):
            if isinstance(msg, AIMessage):
                retrieval_query = f"{msg.content}\n\n{normalized_question}"
                break

        top_hit = vectorstore.similarity_search_with_score(retrieval_query, k=1)
        top_score = top_hit[0][1] if top_hit else float("inf")

        docs = vectorstore.max_marginal_relevance_search(
            retrieval_query, k=TOP_K, fetch_k=FETCH_K, lambda_mult=MMR_LAMBDA
        )

        # A follow-up that names its own fully-specified new topic (e.g. "give me
        # details about the case study on Warfare and weaponisation" right after an
        # answer about a different case study) gets its retrieval hijacked if the
        # prior turn is prepended: the combined embedding skews toward the previous
        # topic and the new topic's own chunks can be pushed out of MMR's top-k
        # entirely (confirmed by testing — a bare search for that question surfaced
        # the correct chunks, but the prior-turn-prepended search returned zero of
        # them). Also searching on the bare current-turn question and merging in any
        # chunks it finds that the prefixed search missed fixes this without
        # regressing genuine pronoun-style follow-ups: their own bare-question
        # retrieval just contributes little of value alongside the prefixed one.
        seen = {d.page_content for d in docs}

        if retrieval_query != normalized_question:
            for d in vectorstore.max_marginal_relevance_search(
                normalized_question, k=TOP_K, fetch_k=FETCH_K, lambda_mult=MMR_LAMBDA
            ):
                if d.page_content not in seen:
                    docs.append(d)
                    seen.add(d.page_content)

        # MMR's diversity trade-off can drop the single most relevant chunk in favor
        # of a more "diverse" one that's actually less useful (confirmed: for one
        # query the true best match ranked #4 by plain similarity was excluded by
        # every MMR parameter combination tried, always swapped for a worse chunk).
        # Plain top-k similarity is a floor under MMR: it guarantees the best raw
        # matches are always present, while MMR's own picks still add extra breadth.
        for doc, _score in vectorstore.similarity_search_with_score(normalized_question, k=TOP_K):
            if doc.page_content not in seen:
                docs.append(doc)
                seen.add(doc.page_content)

        for kw_doc in article_keyword_lookup(normalized_question, vectorstore):
            if kw_doc.page_content not in seen:
                docs.append(kw_doc)
                seen.add(kw_doc.page_content)

        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata['source_name']}]\n{d.page_content}" for d in docs
        )
        sources_retrieved = sorted({d.metadata["source_name"] for d in docs})
        logger.info("retrieve: top_score=%.4f sources=%s", top_score, sources_retrieved)
        return {"context": context, "top_score": top_score, "sources_retrieved": sources_retrieved}

    def generate(state: AgentState) -> dict:
        """Generate a draft response using the LLM with retrieved context."""
        system = SystemMessage(content=SYSTEM_PROMPT.format(context=state["context"]))
        response = llm.invoke([system] + state["messages"])
        return {"draft": response.content}

    def verify(state: AgentState) -> dict:
        """Supervisor check: a deterministic citation check first, then an LLM check for
        faithfulness and relevance if the citation check passes."""
        citation_issue = check_citations(state["draft"], state["sources_retrieved"])
        if citation_issue:
            logger.info("verify: citation check failed: %s", citation_issue)
            return {"verification": f"INVALID: {citation_issue}"}

        question = state["messages"][-1].content
        check = VERIFY_PROMPT.format(question=question, context=state["context"], draft=state["draft"])
        result = verifier_llm.invoke([HumanMessage(content=check)])
        verdict = result.content.strip()
        logger.info("verify: %s", verdict)
        return {"verification": verdict}

    def route_after_verify(state: AgentState) -> Literal["finalize", "revise"]:
        return "finalize" if state["verification"].upper().startswith("VALID") else "revise"

    def revise(state: AgentState) -> dict:
        """Regenerate the answer once, addressing the specific verification failure."""
        question = state["messages"][-1].content
        system = SystemMessage(content=SYSTEM_PROMPT.format(context=state["context"]))
        correction = HumanMessage(
            content=REVISE_PROMPT.format(reason=state["verification"], question=question)
        )
        response = llm.invoke([system] + state["messages"] + [correction])
        return {"draft": response.content}

    def finalize(state: AgentState) -> dict:
        """Commit the (possibly revised) draft as the final answer."""
        return {"messages": [AIMessage(content=state["draft"])]}

    # Build agent graph: retrieve -> generate -> verify -> [finalize | revise -> finalize]
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_node("verify", verify)
    graph_builder.add_node("revise", revise)
    graph_builder.add_node("finalize", finalize)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", "verify")
    graph_builder.add_conditional_edges("verify", route_after_verify)
    graph_builder.add_edge("revise", "finalize")
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile(checkpointer=MemorySaver())


USER_AVATAR = "🧑‍💻"
ASSISTANT_AVATAR = "⚖️"

# EU flag colors (Pantone Reflex Blue #003399, gold #FFCC00) as the base
# palette, rather than a generic purple gradient — the subject is European
# AI regulation, so the theme should read as European at a glance.
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.hero {
    text-align: center;
    padding: 1.75rem 1rem 1.1rem 1rem;
    margin-bottom: 0.25rem;
    border-bottom: 3px solid;
    border-image: linear-gradient(90deg, #003399 0%, #FFCC00 100%) 1;
}
.hero h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
    background: linear-gradient(135deg, #002868 0%, #003399 55%, #0052CC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: #4B5768;
    font-size: 1.02rem;
    max-width: 640px;
    margin: 0 auto;
}
.badge-row {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.pill {
    display: inline-block;
    padding: 0.28rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    background: #EAF0FC;
    color: #003399;
    border: 1px solid #B9CCF0;
}
.pill.gold { background: #FFF9E0; color: #8A6D00; border-color: #FFE58A; }

.status-pill {
    display: inline-block;
    margin-top: 0.5rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.status-verified { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
.status-revised   { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }

[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 0.4rem 0.2rem;
    margin-bottom: 0.35rem;
}

/* Chat input: rounded, EU-blue border, gold glow on focus */
div[data-testid="stChatInput"] {
    border-radius: 14px;
}
div[data-testid="stChatInput"] > div {
    border-radius: 14px;
    border: 1.5px solid #B9CCF0;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}
div[data-testid="stChatInput"]:focus-within > div {
    border-color: #003399;
    box-shadow: 0 0 0 3px rgba(255, 204, 0, 0.35);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F0F4FC 0%, #FFFFFF 50%);
}

.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin-top: 0.5rem;
}
.stat-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 0.7rem 0.8rem;
    border-left: 4px solid var(--accent, #003399);
    box-shadow: 0 1px 3px rgba(15, 26, 60, 0.08);
}
.stat-card .stat-icon { font-size: 1.05rem; }
.stat-card .stat-value { font-size: 1.5rem; font-weight: 800; color: #0F1A3C; line-height: 1.15; }
.stat-card .stat-label { font-size: 0.72rem; color: #6B7690; font-weight: 600; text-transform: uppercase; letter-spacing: 0.02em; }

.reliability-bar-track {
    width: 100%;
    height: 9px;
    background: #E7ECF7;
    border-radius: 999px;
    overflow: hidden;
    margin: 0.75rem 0 0.3rem 0;
}
.reliability-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #003399 0%, #FFCC00 100%);
    border-radius: 999px;
}
.reliability-bar-label {
    font-size: 0.75rem;
    color: #6B7690;
    font-weight: 600;
}

div[data-testid="stButton"] button {
    border-radius: 10px;
    border: 1px solid #B9CCF0;
    background: #FFFFFF;
    color: #003399;
    font-weight: 600;
    transition: all 0.15s ease;
}
div[data-testid="stButton"] button:hover {
    background: #EAF0FC;
    border-color: #003399;
}
</style>
"""


def render_status_badge(verification: str) -> str:
    """Return an HTML pill summarizing the supervisor's verdict for one answer."""
    if verification.upper().startswith("VALID"):
        return '<span class="status-pill status-verified">✅ Verified as grounded in the knowledge base</span>'
    return '<span class="status-pill status-revised">🔁 Revised after failing a groundedness/relevance check</span>'


def ask_agent(agent, question: str) -> None:
    """Run one question through the agent, render it, and update session state."""
    st.chat_message("user", avatar=USER_AVATAR).write(question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("🤔 Checking the EU AI Act, bias research, and trustworthy AI guidelines..."):
            result = agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                config={"configurable": {"thread_id": "streamlit-session"}},
            )
        answer = result["messages"][-1].content
        verification = result.get("verification", "")

        stats = st.session_state.stats
        stats["total"] += 1
        if verification.upper().startswith("VALID"):
            stats["verified"] += 1
        else:
            stats["revised"] += 1
        if result.get("top_score", 0) > LOW_CONFIDENCE_LOG_THRESHOLD:
            stats["low_confidence"] += 1

        st.write(answer)
        st.markdown(render_status_badge(verification), unsafe_allow_html=True)

    st.session_state.history += [
        {"role": "user", "text": question, "badge": None},
        {"role": "assistant", "text": answer, "badge": render_status_badge(verification)},
    ]


def main():
    """Main Streamlit application."""

    # Page configuration
    st.set_page_config(
        page_title="AI Ethics Assistant",
        page_icon="⚖️",
        layout="centered",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <h1>⚖️ AI Ethics Assistant</h1>
            <p>🇪🇺 Expert on European AI ethics & regulation — grounded in the EU AI Act,
            EU ethics guidelines and academic research.</p>
            <div class="badge-row">
                <span class="pill">🧠 LangGraph Agent</span>
                <span class="pill gold">🛡️ Self-Verifying</span>
                <span class="pill">⚡ Groq-Powered</span>
                <span class="pill gold">🆓 Free & Open Source</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load agent with error handling
    try:
        agent = load_agent()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    # Initialize chat history and reliability stats
    if "history" not in st.session_state:
        st.session_state.history = []
    if "stats" not in st.session_state:
        st.session_state.stats = {"total": 0, "verified": 0, "revised": 0, "low_confidence": 0}

    # Display chat history (with each answer's original verification badge)
    for msg in st.session_state.history:
        with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR):
            st.write(msg["text"])
            if msg["badge"]:
                st.markdown(msg["badge"], unsafe_allow_html=True)

    # Handle user input
    if question := st.chat_input("Ask about the EU AI Act, bias, trustworthy AI..."):
        ask_agent(agent, question)

    # Sidebar: live, measurable reliability stats (rather than an anecdotal
    # "it seems fine"), plus a way to start over that the old suggested-
    # question chips didn't offer.
    with st.sidebar:
        st.markdown("### 📊 Session Reliability")
        stats = st.session_state.stats
        if stats["total"]:
            verified_pct = stats["verified"] / stats["total"] * 100
            st.markdown(
                f"""
                <div class="stat-grid">
                    <div class="stat-card" style="--accent:#003399;">
                        <div class="stat-icon">💬</div>
                        <div class="stat-value">{stats['total']}</div>
                        <div class="stat-label">Questions</div>
                    </div>
                    <div class="stat-card" style="--accent:#059669;">
                        <div class="stat-icon">✅</div>
                        <div class="stat-value">{stats['verified']}</div>
                        <div class="stat-label">Verified</div>
                    </div>
                    <div class="stat-card" style="--accent:#D97706;">
                        <div class="stat-icon">🔁</div>
                        <div class="stat-value">{stats['revised']}</div>
                        <div class="stat-label">Revised</div>
                    </div>
                    <div class="stat-card" style="--accent:#6B7690;">
                        <div class="stat-icon">🔎</div>
                        <div class="stat-value">{stats['low_confidence']}</div>
                        <div class="stat-label">Low Confidence</div>
                    </div>
                </div>
                <div class="reliability-bar-label">{verified_pct:.0f}% verified on first try</div>
                <div class="reliability-bar-track">
                    <div class="reliability-bar-fill" style="width:{verified_pct:.0f}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption("Ask a question to see reliability stats for this session.")

        if st.session_state.history:
            st.button(
                "🔄 Start new conversation",
                use_container_width=True,
                on_click=lambda: (
                    st.session_state.update(
                        history=[], stats={"total": 0, "verified": 0, "revised": 0, "low_confidence": 0}
                    )
                ),
            )

        st.divider()
        with st.expander("ℹ️ About this assistant"):
            st.markdown(
                "- **LLM (generation):** Groq (`llama-3.1-8b-instant`)\n"
                "- **LLM (verification):** Groq (`llama-3.3-70b-versatile`)\n"
                "- **Embeddings:** local Sentence Transformers\n"
                "- **Vector store:** ChromaDB (641 chunks)\n"
                "- **Agent:** LangGraph — retrieve → generate → verify → revise\n"
                "- **Sources:** EU AI Act, Bias & Fairness Survey (Mehrabi et al.), "
                "EP Ethics of AI study, EU Trustworthy AI Guidelines"
            )
            st.markdown(
                "[📦 Source on GitHub](https://github.com/Najam0786/AI-ETHICS-ASSISTANT)"
            )


if __name__ == "__main__":
    main()
