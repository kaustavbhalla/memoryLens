"""Mode 1: deterministic pipeline — face → FAISS → SQLite → card."""

import numpy as np

from server.memory.store import memory
from server.hud.renderer import build_person_card, HUDCard, EmptyCard


class DeterministicPipeline:
    """
    Fixed, sequential retrieval. No branching, no LLM, no surprises.
    Called every time a face is confirmed by the fusion engine.

    Sequence (total target: <100ms):
      1. FAISS ANN search          → person_id        (~10ms)
      2. SQLite person lookup      → profile           (~5ms)
      3. SQLite last N summaries   → recent context    (~5ms)
      4. SQLite top facts          → quick facts       (~5ms)
      5. Build HUD card            → rendered output   (~1ms)

    Does NOT touch: LanceDB, NetworkX, Claude API.
    """

    async def run(self, face_embedding: np.ndarray) -> HUDCard:
        # Step 1 — biometric match
        person_id, confidence = memory.biometric.search_face(face_embedding)
        if person_id is None:
            return EmptyCard()

        # Step 2 — person profile
        person = memory.db.get_person(person_id)
        if person is None:
            return EmptyCard()

        # Step 3 — last 2 conversation summaries
        recent = memory.db.get_recent_conversations(person_id, limit=2)

        # Step 4 — top 5 atomic facts by confidence
        facts = memory.db.get_top_facts(person_id, limit=5)

        # Step 5 — assemble card (no LLM, pure template)
        return build_person_card(person, recent, facts, memory.db.get_patient_name())
