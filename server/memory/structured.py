"""SQLite ORM (SQLModel) — Person, Fact, Conversation."""
from sqlmodel import SQLModel, Field, create_engine, Session, desc, select, true, text, and_, func, col
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
    relation_confidence : float = Field(default=1.0)
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

    id:                         str = Field(default_factory=newID, primary_key=True)
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

    id:                     str = Field(default_factory=newID, primary_key=True)
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

    id:             str = Field(default_factory=newID, primary_key=True)
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
    
    def get_top_facts(self, person_id: str, limit: int = 5) -> list[AtomicFact]:
        with Session(self.engine) as s:
            return list(s.exec(
                select(AtomicFact)
                .where(
                    AtomicFact.person_id == person_id,
                    AtomicFact.is_outdated == False
                )
            .order_by(col(AtomicFact.confidence).desc())
            .limit(limit)
            ).all())
    
    def get_all_active_facts(self, person_id: str) -> list[AtomicFact]:
        with Session(self.engine) as s:
            return list(s.exec(
                select(AtomicFact)
                .where(
                    AtomicFact.person_id == person_id,
                    AtomicFact.is_outdated == False
                )
                .order_by(col(AtomicFact.confidence).desc())
            ).all())

    def get_patient_profile(self) -> dict[str, str]:
        with Session(self.engine) as s:
            rows = s.exec(select(PatientProfile)).all()
            return {r.key: r.value for r in rows}
    
    def get_inter_person_relations(self, person_id: str) -> list[PersonRelation]:
        with Session(self.engine) as s:
            return list(s.exec(
                select(PersonRelation)
                .where(PersonRelation.person_a_id == person_id)
                .order_by(col(PersonRelation.confidence).desc())
            ).all())
    
    #MAYBE BROKEN
    def get_shared_connections(self, person_a: str,
                               person_b: str) -> list[PersonRelation]:
        """People that both A and B know. Used in social context for recall."""
        with Session(self.engine) as s:
            a_targets = s.exec(
                select(PersonRelation.person_b_id)
                .where(PersonRelation.person_a_id == person_a)
            ).all()
            b_targets = s.exec(
                select(PersonRelation.person_b_id)
                .where(PersonRelation.person_a_id == person_b)
            ).all()
            shared_ids = set(a_targets) & set(b_targets)
            if not shared_ids:
                return []
            a = list(s.exec(
                select(PersonRelation)
                .where(col(Person.id).in_(shared_ids))
            ).all())

            return a

    # ── Consolidation agent writes ────────────────────────────────────

    def save_person(self, person: Person) -> None:
        with Session(self.engine) as s:
            s.add(person)
            s.commit()
            s.refresh(person)

    def save_conversation(self, conv: Conversation) -> None:
        with Session(self.engine) as s:
            s.add(conv)
            s.commit()

    def save_fact(self, fact: AtomicFact) -> None:
        with Session(self.engine) as s:
            s.add(fact)
            s.commit()

    def mark_fact_outdated(self, fact_id: str,
                           superseded_by_id: str) -> None:
        with Session(self.engine) as s:
            fact = s.get(AtomicFact, fact_id)
            if fact:
                fact.is_outdated = True
                fact.superseded_by = superseded_by_id
                s.add(fact)
                s.commit()

    def update_relationship_summary(self, person_id: str,
                                    summary: str) -> None:
        with Session(self.engine) as s:
            person = s.get(Person, person_id)
            if person:
                person.relationship_summary = summary
                person.relationship_summary_updated = utcnow()
                s.add(person)
                s.commit()

    def mark_conversation_consolidated(self, conversation_id: str) -> None:
        with Session(self.engine) as s:
            conv = s.get(Conversation, conversation_id)
            if conv:
                conv.consolidation_status = ConsolidationStatus.CONSOLIDATED
                s.add(conv)
                s.commit()

    def get_pending_consolidations(self) -> list[Conversation]:
        """
        Startup recovery: find any sessions that weren't consolidated
        (e.g. due to crash or API failure). Requeue them.
        """
        with Session(self.engine) as s:
            return list(s.exec(
                select(Conversation)
                .where(Conversation.consolidation_status ==
                       ConsolidationStatus.PENDING)
                .order_by(col(Conversation.timestamp).asc())
            ).all())

    def upsert_person_relation(self, rel: PersonRelation) -> None:
        """
        Insert or update inter-person relation.
        If same (a, b, type) exists, update confidence if higher.
        """
        with Session(self.engine) as s:
            existing = s.exec(
                select(PersonRelation)
                .where(
                    PersonRelation.person_a_id == rel.person_a_id,
                    PersonRelation.person_b_id == rel.person_b_id,
                    PersonRelation.relation_type == rel.relation_type
                )
            ).first()
            if existing:
                if rel.confidence > existing.confidence:
                    existing.confidence = rel.confidence
                    existing.evidence_text = rel.evidence_text
                    s.add(existing)
            else:
                s.add(rel)
            s.commit()

    # ── Caregiver portal queries ──────────────────────────────────────

    def get_enrollment_queue(self) -> list[Person]:
        """Provisional auto-enrolled profiles awaiting caregiver review."""
        with Session(self.engine) as s:
            return list(s.exec(
                select(Person)
                .where(col(Person.enrollment_status).in_([
                    EnrollmentStatus.AUTO,
                    EnrollmentStatus.FLAGGED
                ]))
                .order_by(col(Person.first_seen).desc())
            ).all())

    def confirm_enrollment(self, person_id: str,
                           confirmed_name: str,
                           confirmed_relation: str) -> None:
        with Session(self.engine) as s:
            person = s.get(Person, person_id)
            if person:
                person.name = confirmed_name
                person.relation = confirmed_relation
                person.name_confidence = 1.0
                person.relation_confidence = 1.0
                person.enrollment_status = EnrollmentStatus.CONFIRMED
                s.add(person)
                s.commit()

    def get_all_persons(self, include_unconfirmed: bool = False) -> list[Person]:
        with Session(self.engine) as s:
            q = select(Person)
            if not include_unconfirmed:
                q = q.where(Person.enrollment_status ==
                             EnrollmentStatus.CONFIRMED)
            return list(s.exec(
                q.order_by(col(Person.last_seen).desc())
            ).all())

    def update_patient_profile(self, key: str, value: str) -> None:
        with Session(self.engine) as s:
            existing = s.get(PatientProfile, key)
            if existing:
                existing.value = value
                existing.last_updated = utcnow()
                s.add(existing)
            else:
                s.add(PatientProfile(key=key, value=value))
            s.commit()
