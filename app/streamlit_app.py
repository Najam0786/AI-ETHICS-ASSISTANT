"""
AI Ethics Assistant - Streamlit Web Interface

A RAG-based conversational agent for European AI ethics and regulation.
Uses Groq for LLM and local Sentence Transformers for embeddings.
"""

import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from sentence_transformers import SentenceTransformer

# Configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

CHROMA_DIR = str(ROOT_DIR / "chroma_db")
COLLECTION_NAME = "ai_ethics_eu"
LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 4

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

VERIFY_PROMPT = """You are a strict compliance reviewer for an AI ethics assistant. Your only \
job is to check whether a DRAFT ANSWER is fully grounded in the given CONTEXT — no invented \
facts, article numbers, or legal claims that aren't directly supported.

CONTEXT:
{context}

DRAFT ANSWER:
{draft}

Reply with exactly "VALID" if every claim in the draft is directly supported by the context \
(or the draft correctly states it doesn't have enough information). Otherwise reply with \
"INVALID: " followed by a one-sentence explanation of the unsupported claim.
"""

REVISE_PROMPT = """Your previous draft failed a groundedness check: {reason}

Rewrite the answer to the question below using ONLY the context provided, correcting this \
issue. If the context doesn't actually support an answer, say so explicitly.

QUESTION: {question}
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    draft: str
    verification: str


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
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    # Initialize LLM
    llm = ChatGroq(model=LLM_MODEL, temperature=0.2)

    # Define agent nodes
    def retrieve(state: AgentState) -> dict:
        """Retrieve relevant documents from vector store."""
        question = state["messages"][-1].content
        docs = retriever.invoke(question)
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata['source_name']}]\n{d.page_content}" for d in docs
        )
        return {"context": context}

    def generate(state: AgentState) -> dict:
        """Generate a draft response using the LLM with retrieved context."""
        system = SystemMessage(content=SYSTEM_PROMPT.format(context=state["context"]))
        response = llm.invoke([system] + state["messages"])
        return {"draft": response.content}

    def verify(state: AgentState) -> dict:
        """Supervisor check: is the draft fully grounded in the retrieved context?"""
        check = VERIFY_PROMPT.format(context=state["context"], draft=state["draft"])
        result = llm.invoke([HumanMessage(content=check)])
        return {"verification": result.content.strip()}

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
        """Commit the (possibly revised) draft as the agent's final answer."""
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

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

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
        verified = result.get("verification", "").upper().startswith("VALID")

        # Display response and update history
        with st.chat_message("assistant"):
            st.write(answer)
            if verified:
                st.caption("✅ Verified as grounded in the knowledge base")
            else:
                st.caption("🔁 Revised after a groundedness check")
        st.session_state.history += [("user", question), ("assistant", answer)]


if __name__ == "__main__":
    main()
