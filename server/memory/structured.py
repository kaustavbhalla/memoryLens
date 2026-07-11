"""SQLite ORM (SQLModel) — Person, Fact, Conversation."""
from sqlmodel import SQLModel, Field, create_engine, Session, select, true, text, and_, func, col
from typing import Optional
from datetime import datetime, timezone
from enum import Enum
import uuid
import json
from pathlib import Path

DATABASE_PATH = Path("data/memoryLens.db")

class EnrollmentStatus(str, Enum):
    CONFIRMED = "confirmed"
    AUTO = "auto"
    FLAGGED = "flagged"

class EmotionalTone(str, Enum):
    WARM = "warm"
    NEUTRAL = "neutral"
    STRESSFUL = "stressful"

class ConsolidationStatus(str, Enum):
    PENDING = "pending"
    CONSOLIDATED = "consolidated"

class RelationType(str, Enum):
    KNOWS = "knows"
    WORKS_WITH = "works_with"
    RELATED_TO = "related_to"
    MENTIONED_TOGETHER = "mentioned_together"

class RelationSource(str, Enum):
    CAREGIVER = "caregiver"
    EXTRACTED = "extracted"


# Helpers 

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def newID() -> str:
    return str(uuid.uuid4())


#Models

class Person(SQLModel, table=True):
    __tablename__ = "person"

    id: str = Field(default_factory=newID, primary_key=True)
    name: str
    relation: str = Field(default="unknown")
    relation_confidence : int = Field(default=1.0)
    first_seen: str = Field(default_factory=utcnow)
    last_seen: str = Field(default_factory=utcnow)
    visit_count: int = Field(default=0)
    emotional_baseline: float = Field(default=0.5)
    face_embedding_id: Optional[str] = None
    voice_embedding_id: Optional[str] = None
    notes: Optional[str] = None
    enrollment_status: EnrollmentStatus = Field(default=EnrollmentStatus.CONFIRMED)
    name_confidence: float = Field(default=1.0)
    relationship_summary: Optional[str] = None
    relationship_summary_updated: Optional[str] = None

    @property
    def display_name(self) -> str:
        if self.name_confidence >= 0.75:
            return self.name
        return "Unknown person"
    
    @property
    def hud_summary(self) -> str:
        if self.relationship_summary:
            s = self.relationship_summary
            if len(s) > 120:
                return s[120] + "..."
            else:
                return s
        return "No history yet"
    
    def mark_seen(self) ->None:
        self.last_seen = utcnow()
        self.visit_count += 1

class Conversation(SQLModel, table=True):
    __tablename__ = "conversation"

    id:                         str = Field(default_factory=new_id, primary_key=True)
    person_id:                  str = Field(foreign_key="person.id")
    timestamp:                  str = Field(default_factory=utcnow)
    duration_seconds:           int = Field(default=0)
    summary:                    str = Field(default="")
    emotional_tone:             EmotionalTone = Field(default=EmotionalTone.NEUTRAL)
    key_topics:                 str = Field(default="[]")   # JSON
    raw_transcript_path:        Optional[str] = None
    consolidation_status:       ConsolidationStatus = Field(default=ConsolidationStatus.PENDING)
    consolidation_attempted_at: Optional[str] = None

    def get_topics(self) -> list[str]:
        return json.loads(self.key_topics)

    def set_topics(self, topics: list[str]) -> None:
        self.key_topics = json.dumps(topics)


class AtomicFact(SQLModel, table=True):
    __tablename__ = "atomic_fact"

    id:                     str = Field(default_factory=new_id, primary_key=True)
    person_id:              str = Field(foreign_key="person.id")
    fact_text:              str
    confidence:             float = Field(default=0.8)
    source_conversation_id: Optional[str] = Field(default=None,
                                foreign_key="conversation.id")
    timestamp:              str = Field(default_factory=utcnow)
    is_outdated:            bool = Field(default=False)
    superseded_by:          Optional[str] = Field(default=None,
                                foreign_key="atomic_fact.id")

class PersonRelation(SQLModel, table=True):
    __tablename__ = "person_relation"

    id:             str = Field(default_factory=new_id, primary_key=True)
    person_a_id:    str = Field(foreign_key="person.id")
    person_b_id:    str = Field(foreign_key="person.id")
    relation_type:  RelationType
    confidence:     float = Field(default=0.8)
    source:         RelationSource = Field(default=RelationSource.EXTRACTED)
    evidence_text:  Optional[str] = None
    extracted_at:   str = Field(default_factory=utcnow)

class PatientProfile(SQLModel, table=True):
    __tablename__ = "patient_profile"

    key:          str = Field(primary_key=True)
    value:        str
    last_updated: str = Field(default_factory=utcnow)


#init db

def get_engine():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{DATABASE_PATH}",
        echo=False,
        connect_args={
            "check_same_thread": False,
            "timeout": 10
        }
    )
    return engine

def init_db(engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode = WAL"))
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.execute(text("PRAGMA synchronous = NORMAL"))
        conn.execute(text("PRAGMA cache_size = -64000"))
        conn.execute(text("PRAGMA temp_store = MEMORY"))

    SQLModel.metadata.create_all(engine)

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_person_enrollment_status
                ON person(enrollment_status)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_person_last_seen
                ON person(last_seen DESC)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conversation_person_timestamp
                ON conversation(person_id, timestamp DESC)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conversation_consolidation_status
                ON conversation(consolidation_status)
                WHERE consolidation_status = 'pending'
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_atomic_fact_person_active
                ON atomic_fact(person_id, confidence DESC)
                WHERE is_outdated = 0
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_atomic_fact_person_all
                ON atomic_fact(person_id, timestamp DESC)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_person_relation_a
                ON person_relation(person_a_id, confidence DESC)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_person_relation_b
                ON person_relation(person_b_id)
        """))
        conn.commit()

class StructuredStore:
    def __init__(self, engine):
        self.engine = engine
        
    def get_person(self, person_id: str) -> Person | None:
        with Session(self.engine) as s:
            return s.get(Person, person_id)
    
    def get_recent_conversations(self, person_id: str, limit: int = 2) -> list[Conversation]:
        """ Last N Conversations"""
        with Session(self.engine) as s:
            return list(s.exec(
                select(Conversation)
                .where(Conversation.person_id == person_id)
                .order_by(col(Conversation.timestamp).desc())
                .limit(limit)
            ))
