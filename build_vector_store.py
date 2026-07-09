"""
Vector Store Builder for AI Ethics Assistant

Processes PDF documents and creates a ChromaDB vector store using local
Sentence Transformers embeddings. No API keys or rate limits required.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

# Configuration
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "ai_ethics_eu"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 150

# Document source mapping
SOURCES = {
    "EU-AI-Act.pdf": "EU AI Act",
    "bias_fairness_survey.pdf": "Bias & Fairness Survey (Mehrabi et al.)",
    "ethics_of_ai_study.pdf": "EP Study: Ethics of AI",
    "trustworthy_ai_guidelines.pdf": "Ethics Guidelines for Trustworthy AI",
}

# Citation-list text (author names, paper titles, years) shares surface vocabulary
# with real questions — e.g. bibliography entries mentioning "autonomous vehicles"
# or "AI" — without containing any substantive answer. Testing found MMR retrieval
# repeatedly pulling in bibliography chunks as "diverse" alternatives ahead of the
# actually relevant chunk, degrading answer quality. Three of the four source PDFs
# have a References/Bibliography section (up to ~20% of total pages); it's dropped
# at ingestion so it never enters the vector store.
REFERENCES_HEADING_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)


def truncate_at_references(pages: list) -> list:
    """Cut a document's pages at its References/Bibliography heading, if any."""
    for i, page in enumerate(pages):
        match = REFERENCES_HEADING_RE.search(page.page_content)
        if match:
            page.page_content = page.page_content[: match.start()]
            return pages[: i + 1]
    return pages


class LocalEmbeddings:
    """Local sentence-transformers embeddings wrapper for LangChain compatibility."""
    
    def __init__(self, model_name: str):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=True).tolist()
    
    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, show_progress_bar=False).tolist()


def clean_text(text: str) -> str:
    """Clean PDF-extracted text by normalizing whitespace and fixing hyphenation."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"-\s+", "", text)
    return text.strip()


def load_documents() -> list:
    """Load and clean PDF documents from the data directory."""
    print("Loading PDF documents...")
    documents = []
    
    for filename, source_name in SOURCES.items():
        path = os.path.join(DATA_DIR, filename)
        loader = PyPDFLoader(path)
        pages = truncate_at_references(loader.load())

        for page in pages:
            page.page_content = clean_text(page.page_content)
            page.metadata["source_name"] = source_name
        
        documents.extend(pages)
        print(f"  Loaded {len(pages):>3} pages ← {source_name}")
    
    print(f"Total pages loaded: {len(documents)}")
    return documents


def split_documents(documents: list) -> list:
    """Split documents into chunks for vector storage."""
    print("\nSplitting documents into chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """Create ChromaDB vector store with local embeddings."""
    print("\nCreating embeddings and indexing to ChromaDB...")
    print("Using local sentence-transformers (no API limits)")
    
    embeddings = LocalEmbeddings(EMBEDDING_MODEL)
    
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    
    print(f"Indexing {len(chunks)} chunks...")
    vectorstore.add_documents(chunks)
    
    return vectorstore


def main():
    """Main execution function."""
    print("=" * 60)
    print("AI Ethics Assistant - Vector Store Builder")
    print("=" * 60)
    
    # Load and process documents
    documents = load_documents()
    chunks = split_documents(documents)
    
    # Create vector store
    vectorstore = create_vector_store(chunks)
    
    # Report results
    print(f"\n{'=' * 60}")
    print(f"✅ Successfully indexed {vectorstore._collection.count()} chunks")
    print(f"✅ Vector store created at: {CHROMA_DIR}")
    print(f"{'=' * 60}")
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
