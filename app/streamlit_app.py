"""
AI Ethics Assistant - Streamlit Web Interface

A RAG-based conversational agent for European AI ethics and regulation.
Uses Groq for LLM and local Sentence Transformers for embeddings.
"""

import logging
import os
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

import streamlit as st
from dotenv import load_dotenv
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
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4
FETCH_K = 20            # candidates considered by MMR before diversity re-ranking
MMR_LAMBDA = 0.5        # 0 = max diversity, 1 = pure relevance

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

    # Initialize LLM
    llm = ChatGroq(model=LLM_MODEL, temperature=0.2)

    # Define agent nodes
    def retrieve(state: AgentState) -> dict:
        """Retrieve relevant documents, using MMR for diversity and plain similarity for a
        confidence score (MMR re-ranks for diversity, so its distances aren't a clean
        relevance signal — a separate top-1 lookup gives us that)."""
        question = state["messages"][-1].content

        # A bare follow-up like "give an example of it" has no semantic content on its
        # own and always scores poorly, which would wrongly trigger abstention on every
        # pronoun-referencing follow-up. Prepend the prior assistant answer (if any) to
        # give retrieval enough context to resolve it — generate() still sees the raw
        # question via the full message history, so this only affects retrieval.
        retrieval_query = question
        for msg in reversed(state["messages"][:-1]):
            if isinstance(msg, AIMessage):
                retrieval_query = f"{msg.content}\n\n{question}"
                break

        top_hit = vectorstore.similarity_search_with_score(retrieval_query, k=1)
        top_score = top_hit[0][1] if top_hit else float("inf")

        docs = vectorstore.max_marginal_relevance_search(
            retrieval_query, k=TOP_K, fetch_k=FETCH_K, lambda_mult=MMR_LAMBDA
        )
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
        result = llm.invoke([HumanMessage(content=check)])
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


def main():
    """Main Streamlit application."""

    # Page configuration
    st.set_page_config(
        page_title="AI Ethics Assistant",
        page_icon="⚖️",
        layout="centered"
    )

    st.title("⚖️ AI Ethics Assistant")
    st.caption(
        "Expert on European AI ethics & regulation — grounded in the EU AI Act, "
        "EU ethics guidelines and academic research."
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

    # Display chat history
    for role, text in st.session_state.history:
        st.chat_message(role).write(text)

    # Handle user input
    if question := st.chat_input("Ask about the EU AI Act, bias, trustworthy AI..."):
        st.chat_message("user").write(question)

        # Generate response
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": "streamlit-session"}},
        )
        answer = result["messages"][-1].content
        verification = result.get("verification", "").upper()

        stats = st.session_state.stats
        stats["total"] += 1
        if verification.startswith("VALID"):
            stats["verified"] += 1
            badge = "✅ Verified as grounded in the knowledge base"
        else:
            stats["revised"] += 1
            badge = "🔁 Revised after failing a groundedness/relevance check"
        if result.get("top_score", 0) > LOW_CONFIDENCE_LOG_THRESHOLD:
            stats["low_confidence"] += 1

        # Display response and update history
        with st.chat_message("assistant"):
            st.write(answer)
            st.caption(badge)
        st.session_state.history += [("user", question), ("assistant", answer)]

    # Session reliability stats — a live, measurable view of how often the
    # supervisor had to intervene, rather than an anecdotal "it seems fine".
    with st.sidebar:
        st.subheader("📊 Session Reliability")
        stats = st.session_state.stats
        if stats["total"]:
            st.metric("Questions asked", stats["total"])
            st.metric("Verified first try", f"{stats['verified']} ({stats['verified'] / stats['total'] * 100:.0f}%)")
            st.metric("Revised after check", stats["revised"])
            st.metric("Low retrieval confidence", stats["low_confidence"])
        else:
            st.caption("Ask a question to see reliability stats for this session.")


if __name__ == "__main__":
    main()
