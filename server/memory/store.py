"""Unified memory interface (coordinates all stores)."""

from pathlib import Path
from server.config import (
    DB_PATH, FAISS_DIR, LANCEDB_DIR, GRAPH_PATH,
)
from server.memory.structured import get_engine, init_db, StructuredStore
from server.memory.biometric import BiometricStore
from server.memory.vector import ConversationVectorStore
from server.memory.graph import RelationshipGraph


class MemoryStore:
    """
    Single entry point that owns all four stores.
    Call load() at startup, save() on shutdown.
    """

    def __init__(self):
        self.db: StructuredStore | None = None
        self.biometric: BiometricStore | None = None
        self.vector: ConversationVectorStore | None = None
        self.graph: RelationshipGraph | None = None

    def load(self):
        # Ensure data dirs exist
        for d in [DB_PATH.parent, FAISS_DIR, LANCEDB_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # Structured store (SQLite)
        engine = get_engine()
        init_db(engine)
        self.db = StructuredStore(engine)

        # Biometric store (FAISS)
        self.biometric = BiometricStore(FAISS_DIR)
        self.biometric.load()

        # Vector store (LanceDB)
        self.vector = ConversationVectorStore(str(LANCEDB_DIR))

        # Relationship graph (NetworkX)
        self.graph = RelationshipGraph(GRAPH_PATH)
        if GRAPH_PATH.exists():
            self.graph.load()

    def save(self):
        """Persist all stores that have disk state."""
        if self.biometric:
            self.biometric.save()
        if self.graph:
            self.graph.save()


# Module-level singleton
memory = MemoryStore()
