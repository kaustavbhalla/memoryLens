"""Env vars, constants, latency budgets, thresholds."""

from pathlib import Path
import os

# ── Paths ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "memoryLens.db"
FAISS_DIR = DATA_DIR / "faiss"
LANCEDB_DIR = DATA_DIR / "lancedb"
GRAPH_PATH = DATA_DIR / "relationship_graph.pkl"

# ── Face detection ────────────────────────────────────────────────────

FACE_DETECTION_CONFIDENCE = 0.6
FACE_RECOGNITION_THRESHOLD = 0.6   # cosine similarity (FAISS inner product)
FACE_REEMBED_INTERVAL_S = 2.0      # re-extract embedding if not confirmed for this long

# ── Voice detection ───────────────────────────────────────────────────

VOICE_RECOGNITION_THRESHOLD = 0.75

# ── Identity fusion ───────────────────────────────────────────────────

FUSION_CONFIRMATION_THRESHOLD = 0.65
FACE_WEIGHT = 0.6
VOICE_WEIGHT = 0.4

# ── Trigger detector ──────────────────────────────────────────────────

TRIGGER_COOLDOWN_S = 30.0

# ── HUD card ──────────────────────────────────────────────────────────

HUD_SUMMARY_MAX_CHARS = 120
HUD_PERSON_CARD_MAX_CHARS = 80

# ── Latency budgets (ms) ─────────────────────────────────────────────

LATENCY_BUDGET_MODE1_MS = 500     # known face → HUD card
LATENCY_BUDGET_MODE2_MS = 3000    # confusion → recall narration
LATENCY_BUDGET_MODE3_MS = 5000    # unknown face → provisional profile

# ── Auto-enrollment quality thresholds ────────────────────────────────

AUTO_ENROLL_NAME_CONFIDENCE_SHOW = 0.75   # above: show name on HUD
AUTO_ENROLL_NAME_CONFIDENCE_STORE = 0.5   # below: store as "Unknown Person"
AUTO_ENROLL_RELATION_CONFIDENCE_MIN = 0.4
AUTO_ENROLL_MIN_FACE_PX = 80
AUTO_ENROLL_MIN_AUDIO_S = 8.0
AUTO_ENROLL_MAX_AGENT_STEPS = 5

# ── Consolidation ─────────────────────────────────────────────────────

CONSOLIDATION_MAX_AGENT_STEPS = 7
RELATIONSHIP_SUMMARY_TARGET_SENTENCES = (4, 6)

# ── Webcam client ─────────────────────────────────────────────────────

WEBCAM_SERVER_URL = os.getenv("MEMORYLENS_SERVER_URL", "http://127.0.0.1:8000")
WEBCAM_INDEX = int(os.getenv("MEMORYLENS_WEBCAM_INDEX", "0"))
WEBCAM_FRAME_WIDTH = 640
WEBCAM_FRAME_HEIGHT = 480
WEBCAM_FPS = 15
WEBCAM_SEND_INTERVAL_S = 0.33   # ~3 FPS to server (avoid flooding)

# ── LLM ───────────────────────────────────────────────────────────────

ANTHROPIC_MODEL = "claude-sonnet-4-6"
ANTHROPIC_MAX_TOKENS_RECALL = 1024
ANTHROPIC_MAX_TOKENS_ENROLL = 512
ANTHROPIC_MAX_TOKENS_EXTRACT = 1000
ANTHROPIC_MAX_TOKENS_NARRATE = 300
