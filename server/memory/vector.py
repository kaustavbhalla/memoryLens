"""LanceDB conversation RAG."""
import lancedb
import numpy as np
import pyarrow as pa
from sentence_transformers import SentenceTransformer

class ConversationVectorStore:
    def __init__(self, db_path: str):
        self.db = lancedb.connect(db_path)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._init_tables()

    
    def _init_tables(self):
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("person_id", pa.string()),
            pa.field("conversation_id", pa.string()),
            pa.field("chunk_text", pa.string()),
            pa.field("timestamp", pa.string()),
            pa.field("chunk_type", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 384)),
        ])

        if "chunks" not in self.db.table_names():
            self.db.create_table("chunks", schema=schema)

        self.table = self.db.open_table("chunks")


    def add_chunks(self, chunks: list[dict]):
        texts = [c["chunk_text"] for c in chunks]
        embeddings = self.embedder.encode(texts, batch_size=32)
        rows = []

        for chunk, emb in zip(chunks, embeddings):
            rows.append({**chunk, "vector": emb.tolist()})

        self.table.add(rows)

    def search(self, query: str, person_id: str, top_k: int = 5) -> list[dict]:
        query_vec = self.embedder.encode([query])[0]
        results = (
            self.table
            .search(query_vec)
            .where(f"person_id = '{person_id}'")
            .limit(top_k)
            .to_list()
        )

        return results
