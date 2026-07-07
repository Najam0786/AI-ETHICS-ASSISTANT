import os
from pathlib import Path
from typing import Annotated, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

CHROMA_DIR = str(ROOT_DIR / "chroma_db")
COLLECTION_NAME = "ai_ethics_eu"

LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
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


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str


st.set_page_config(page_title="AI Ethics Assistant 🇪🇺", page_icon="⚖️")
st.title("⚖️ AI Ethics Assistant")
st.caption(
    "Expert on European AI ethics & regulation — grounded in the EU AI Act, "
    "EU ethics guidelines and academic research."
)


@st.cache_resource
def load_agent():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not found — check your .env file")
    if not Path(CHROMA_DIR).exists():
        raise RuntimeError(
            "Vector store not found. Run the notebook (notebooks/ai_ethics_assistant.ipynb) "
            "once first to build chroma_db/."
        )

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)

    def retrieve(state: AgentState) -> dict:
        question = state["messages"][-1].content
        docs = retriever.invoke(question)
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata['source_name']}]\n{d.page_content}" for d in docs
        )
        return {"context": context}

    def generate(state: AgentState) -> dict:
        system = SystemMessage(content=SYSTEM_PROMPT.format(context=state["context"]))
        response = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile(checkpointer=MemorySaver())


try:
    agent = load_agent()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

if question := st.chat_input("Ask about the EU AI Act, bias, trustworthy AI..."):
    st.chat_message("user").write(question)
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": "streamlit-session"}},
    )
    answer = result["messages"][-1].content
    st.chat_message("assistant").write(answer)
    st.session_state.history += [("user", question), ("assistant", answer)]
