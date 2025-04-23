from langchain.document_loaders import CSVLoader, UnstructuredMarkdownLoader, JSONLoader
from langchain.schema import Document
from instructor_embedded import InstructorEmbedding
from chromadb import Client
from chromadb.config import Settings
import os
from typing import List, Dict
from crewai import Agent

class RAGToolkit:
    """Toolkit for RAG operations: document loading, embedding, and vector search."""
    
    def __init__(self, collection_name: str = "rag_collection"):
        """Initialize RAG toolkit with ChromaDB and embeddings."""
        self.embedding_model = InstructorEmbedding(model_name="hkunlp/instructor-large")
        self.chroma_client = Client(Settings())
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        self.data_dir = "data/"

    def load_documents(self, file_path: str) -> List[Document]:
        """Load documents from .csv, .md, or .json files."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == ".csv":
            loader = CSVLoader(file_path)
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        elif ext == ".json":
            loader = JSONLoader(file_path, jq_schema=".", text_content=False)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        return loader.load()

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        texts = [doc.page_content for doc in documents]
        return self.embedding_model.encode(texts).tolist()

    def store_documents(self, documents: List[Document]):
        """Store documents and their embeddings in ChromaDB."""
        embeddings = self.embed_documents(documents)
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            documents=[doc.page_content for doc in documents]
        )

    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Perform vector search in ChromaDB."""
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

# Agent Definitions (to be moved to src/agents/ in Phase 3)
class QueryBuilderAgent(Agent):
    """Agent to process user queries and generate variants."""
    def __init__(self):
        super().__init__(
            role="Query Builder",
            goal="Understand user query intent and generate query variants",
            backstory="Expert in natural language processing and query optimization.",
            tools=[],
            verbose=True
        )

    def execute(self, query: str) -> Dict:
        """Process query and return variants (placeholder)."""
        return {"original": query, "variants": [query]}

class RetrieverAgent(Agent):
    """Agent to retrieve relevant document chunks from ChromaDB."""
    def __init__(self):
        super().__init__(
            role="Retriever",
            goal="Fetch relevant document chunks based on query",
            backstory="Skilled in vector search and information retrieval.",
            tools=[RAGToolkit()],
            verbose=True
        )

    def execute(self, query_data: Dict) -> List[Dict]:
        """Perform vector search (placeholder)."""
        toolkit = self.tools[0]
        return toolkit.search_documents(query_data["original"])

class ReviewerAgent(Agent):
    """Agent to review retrieved chunks for relevance."""
    def __init__(self):
        super().__init__(
            role="Reviewer",
            goal="Evaluate relevance of retrieved document chunks",
            backstory="Expert in quality assurance and relevance scoring.",
            tools=[],
            verbose=True
        )

    def execute(self, retrieved_chunks: List[Dict]) -> List[Dict]:
        """Filter relevant chunks (placeholder)."""
        return retrieved_chunks  # Basic pass-through for now

class DeciderAgent(Agent):
    """Agent to select the best answer from retrieved chunks."""
    def __init__(self):
        super().__init__(
            role="Decider",
            goal="Rank and select the best answer from retrieved chunks",
            backstory="Skilled in decision-making and answer synthesis.",
            tools=[],
            verbose=True
        )

    def execute(self, reviewed_chunks: List[Dict]) -> str:
        """Select best answer (placeholder)."""
        return reviewed_chunks[0]["text"] if reviewed_chunks else "No answer found."