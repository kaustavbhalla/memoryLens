"""
MemoryLens Webcam Overlay Client

Captures webcam frames + mic audio, streams to the FastAPI server,
and renders HUD cards as OpenCV overlays. Fully automatic — no buttons
required for normal operation.

Usage:
    python -m client.webcam_client [server_url]

Debug overrides (hold key during operation):
    e  — manually enroll the currently visible person
    r  — manually trigger recall with typed phrase
    q  — quit
"""

import base64
import sys
import time
import threading
import queue

import cv2
import httpx
import numpy as np

# ── Config ────────────────────────────────────────────────────────────

SERVER_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
WEBCAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
FRAME_SEND_INTERVAL = 0.33   # ~3 FPS to server

# Audio
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SECONDS = 10     # send 10s chunks to /audio
AUDIO_DTYPE = "int16"

# ── Colors (BGR) ─────────────────────────────────────────────────────

COLOR_KNOWN = (0, 200, 0)
COLOR_AUTO = (0, 180, 255)
COLOR_RECALL = (255, 100, 0)
COLOR_UNKNOWN = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (30, 30, 30)
COLOR_TRANSCRIPT = (180, 180, 180)

# ── Font ──────────────────────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_NAME = 0.7
FONT_SCALE = 0.45
FONT_THICK = 2
FONT_THIN = 1


# ── Audio Capture Thread ──────────────────────────────────────────────

class AudioCapture:
    """Continuously captures mic audio and sends chunks to the server."""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self.last_transcript: str = ""
        self.last_speaker_texts: dict[str, str] = {}
        self.trigger_detected: bool = False
        self._lock = threading.Lock()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._sender_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._sender_thread.start()

    def stop(self):
        self._running = False

    def _capture_loop(self):
        """Capture audio from mic into chunks."""
        try:
            import sounddevice as sd
        except ImportError:
            print("WARNING: sounddevice not installed. Audio capture disabled.")
            print("  Install: uv add sounddevice")
            return

        chunk_samples = AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SECONDS
        buffer = np.zeros(chunk_samples, dtype=np.int16)
        write_pos = 0

        def callback(indata, frames, time_info, status):
            nonlocal buffer, write_pos
            if status:
                pass  # ignore overflow warnings
            chunk = indata[:, 0].copy()  # mono
            remaining = chunk_samples - write_pos
            if remaining > 0:
                take = min(remaining, len(chunk))
                buffer[write_pos:write_pos + take] = chunk[:take]
                write_pos += take

        try:
            with sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                dtype=AUDIO_DTYPE,
                blocksize=int(AUDIO_SAMPLE_RATE * 0.1),  # 100ms blocks
                callback=callback,
            ):
                while self._running:
                    if write_pos >= chunk_samples:
                        self._audio_queue.put(buffer.copy())
                        write_pos = 0
        except Exception as e:
            print(f"Audio capture error: {e}")

    def _send_loop(self):
        """Send audio chunks to server, update transcript state."""
        http = httpx.Client(base_url=self.server_url, timeout=30.0)
        while self._running:
            try:
                audio_chunk = self._audio_queue.get(timeout=1.0)
            except queue.Queue.Empty:
                continue

            try:
                b64 = base64.b64encode(audio_chunk.tobytes()).decode()
                resp = http.post("/audio", json={
                    "audio": b64,
                    "sample_rate": AUDIO_SAMPLE_RATE,
                })
                data = resp.json()
                with self._lock:
                    self.last_transcript = data.get("full_text", "")
                    self.last_speaker_texts = data.get("speaker_texts", {})
                    self.trigger_detected = data.get("trigger_detected", False)
            except Exception:
                pass

    def get_state(self) -> dict:
        with self._lock:
            return {
                "transcript": self.last_transcript,
                "speaker_texts": self.last_speaker_texts.copy(),
                "trigger": self.trigger_detected,
            }


# ── Drawing Helpers ───────────────────────────────────────────────────

def encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("utf-8")


def wrap_text(text: str, max_width: int, font, scale, thickness) -> list[str]:
    words = text.split()
    lines, line = [], ""
    for word in words:
        test = f"{line} {word}".strip()
        (w, _), _ = cv2.getTextSize(test, font, scale, thickness)
        if w > max_width and line:
            lines.append(line)
            line = word
        else:
            line = test
    if line:
        lines.append(line)
    return lines


def draw_person_card(frame: np.ndarray, card: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    status = card.get("enrollment_status", "confirmed")
    border_color = COLOR_KNOWN if status == "confirmed" else COLOR_AUTO
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 3)

    card_w, card_h = 340, 170
    cv2.rectangle(frame, (10, 10), (10 + card_w, 10 + card_h), COLOR_BG, -1)

    y = 30
    cv2.putText(frame, card.get("name", "?"), (20, y),
                FONT, FONT_SCALE_NAME, COLOR_TEXT, FONT_THICK)
    y += 25
    relation_line = f"{card.get('relation', '')}  |  {card.get('last_seen', '')}"
    cv2.putText(frame, relation_line, (20, y),
                FONT, FONT_SCALE, (200, 200, 200), FONT_THIN)
    y += 5
    cv2.line(frame, (20, y), (card_w + 10, y), (80, 80, 80), 1)
    y += 18
    summary = card.get("summary", "")
    for line in wrap_text(summary, card_w - 10, FONT, FONT_SCALE, FONT_THIN)[:4]:
        cv2.putText(frame, line, (20, y), FONT, FONT_SCALE, COLOR_TRANSCRIPT, FONT_THIN)
        y += 18
    return frame


def draw_recall_card(frame: np.ndarray, card: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    narration = card.get("narration", "")
    bar_h = 130
    cv2.rectangle(frame, (0, h - bar_h), (w, h), COLOR_BG, -1)
    cv2.rectangle(frame, (0, h - bar_h), (w, h), COLOR_RECALL, 2)
    cv2.putText(frame, "RECALL", (15, h - bar_h + 22),
                FONT, 0.5, COLOR_RECALL, FONT_THIN)
    lines = wrap_text(narration, w - 30, FONT, FONT_SCALE, FONT_THIN)
    y = h - bar_h + 42
    for line in lines[:4]:
        cv2.putText(frame, line, (15, y), FONT, FONT_SCALE, COLOR_TEXT, FONT_THIN)
        y += 20
    return frame


def draw_transcript_bar(frame: np.ndarray, audio_state: dict) -> np.ndarray:
    """Show live transcript at the bottom of the frame."""
    h, w = frame.shape[:2]
    transcript = audio_state.get("transcript", "")
    if not transcript:
        return frame

    bar_h = 60
    overlay = frame[h - bar_h:h, :].copy()
    cv2.rectangle(frame, (0, h - bar_h), (w, h), COLOR_BG, -1)
    cv2.putText(frame, "LIVE", (10, h - bar_h + 18),
                FONT, 0.35, COLOR_RECALL, FONT_THIN)

    lines = wrap_text(transcript, w - 20, FONT, 0.38, FONT_THIN)
    y = h - bar_h + 35
    for line in lines[-2:]:  # show last 2 lines
        cv2.putText(frame, line, (10, y), FONT, 0.38, COLOR_TRANSCRIPT, FONT_THIN)
        y += 16
    return frame


def draw_status_bar(frame: np.ndarray, fps: float, audio_state: dict) -> np.ndarray:
    h, w = frame.shape[:2]
    has_audio = bool(audio_state.get("transcript"))
    audio_icon = "MIC ON" if has_audio else "MIC OFF"
    status_text = f"FPS: {fps:.0f}  |  {audio_icon}"
    (tw, _), _ = cv2.getTextSize(status_text, FONT, 0.4, 1)
    cv2.putText(frame, status_text, (w - tw - 15, 25),
                FONT, 0.4, (100, 100, 100), 1)
    hint = "e=enroll  r=recall  q=quit"
    (hw, _), _ = cv2.getTextSize(hint, FONT, 0.35, 1)
    cv2.putText(frame, hint, (w - hw - 15, 45), FONT, 0.35, (80, 80, 80), 1)
    return frame


# ── Dialogs (debug override) ──────────────────────────────────────────

def enrollment_dialog(frame: np.ndarray) -> dict | None:
    name, relation, step = "", "", "name"
    while True:
        display = frame.copy()
        h, w = display.shape[:2]
        cv2.rectangle(display, (50, 100), (w - 50, 300), COLOR_BG, -1)
        cv2.rectangle(display, (50, 100), (w - 50, 300), COLOR_AUTO, 2)
        cv2.putText(display, "ENROLL NEW PERSON", (70, 135),
                    FONT, 0.6, COLOR_AUTO, FONT_THICK)
        if step == "name":
            cv2.putText(display, f"Name: {name}_", (70, 175),
                        FONT, 0.55, COLOR_TEXT, FONT_THIN)
            cv2.putText(display, "(type name, Enter to confirm)", (70, 210),
                        FONT, 0.4, (150, 150, 150), 1)
        else:
            cv2.putText(display, f"Name: {name}", (70, 175),
                        FONT, 0.55, COLOR_TEXT, FONT_THIN)
            cv2.putText(display, f"Relation: {relation}_", (70, 210),
                        FONT, 0.55, COLOR_TEXT, FONT_THIN)
            cv2.putText(display, "(daughter, doctor, friend, etc.)", (70, 245),
                        FONT, 0.4, (150, 150, 150), 1)
        cv2.imshow("MemoryLens", display)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            return None
        elif key == 13:
            if step == "name" and name:
                step = "relation"
            elif step == "relation":
                return {"name": name, "relation": relation or "unknown"}
        elif key == 8:
            if step == "name":
                name = name[:-1]
            else:
                relation = relation[:-1]
        elif 32 <= key <= 126:
            if step == "name":
                name += chr(key)
            else:
                relation += chr(key)


def recall_dialog(frame: np.ndarray) -> str | None:
    text = ""
    while True:
        display = frame.copy()
        h, w = display.shape[:2]
        cv2.rectangle(display, (50, 150), (w - 50, 260), COLOR_BG, -1)
        cv2.rectangle(display, (50, 150), (w - 50, 260), COLOR_RECALL, 2)
        cv2.putText(display, "RECALL", (70, 185),
                    FONT, 0.6, COLOR_RECALL, FONT_THICK)
        cv2.putText(display, f'"{text}_"', (70, 220),
                    FONT, 0.5, COLOR_TEXT, FONT_THIN)
        cv2.imshow("MemoryLens", display)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            return None
        elif key == 13 and text:
            return text
        elif key == 8:
            text = text[:-1]
        elif 32 <= key <= 126:
            text += chr(key)


# ── Main Loop ─────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open webcam {WEBCAM_INDEX}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    http = httpx.Client(base_url=SERVER_URL, timeout=10.0)

    # Start audio capture (automatic)
    audio = AudioCapture(SERVER_URL)
    audio.start()

    last_card = None
    last_mode = "idle"
    frame_count = 0
    fps_timer = time.time()
    fps = 0.0
    last_recall_time = 0.0
    RECALL_COOLDOWN = 30.0  # don't spam recall

    print(f"MemoryLens connected to {SERVER_URL}")
    print("Audio capture started. System is fully automatic.")
    print("Debug: e=enroll  r=manual recall  q=quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS
            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = frame_count / (time.time() - fps_timer)
                frame_count = 0
                fps_timer = time.time()

            # ── AUTOMATIC RECALL: check if trigger was detected from audio ──
            audio_state = audio.get_state()
            now = time.time()
            if audio_state["trigger"] and (now - last_recall_time) > RECALL_COOLDOWN:
                last_recall_time = now
                # Auto-trigger recall with transcript as context
                try:
                    # Find confirmed person from last card
                    person_id = None
                    if last_card and last_card.get("type") == "person":
                        person_id = last_card.get("person_id")

                    resp = http.post("/recall", json={
                        "trigger_phrase": audio_state["transcript"],
                        "confirmed_person_id": person_id,
                        "session_context": audio_state["transcript"],
                    })
                    last_card = resp.json()
                    last_mode = "recall"
                except Exception as e:
                    print(f"Auto-recall failed: {e}")

            # ── Debug overrides (hold key) ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("e"):
                result = enrollment_dialog(frame)
                if result:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    b64 = base64.b64encode(buf).decode()
                    try:
                        resp = http.post("/enroll", json={
                            "name": result["name"],
                            "relation": result["relation"],
                            "image": b64,
                        })
                        data = resp.json()
                        if "error" in data:
                            print(f"Enroll error: {data['error']}")
                        else:
                            print(f"Enrolled: {data['name']} ({data['relation']})")
                    except Exception as e:
                        print(f"Enroll failed: {e}")
                    continue
            elif key == ord("r"):
                phrase = recall_dialog(frame)
                if phrase:
                    try:
                        resp = http.post("/recall", json={
                            "trigger_phrase": phrase,
                            "session_context": audio_state.get("transcript", ""),
                        })
                        last_card = resp.json()
                        last_mode = "recall"
                    except Exception as e:
                        print(f"Manual recall failed: {e}")
                    continue

            # ── Send frame to server (throttled) ──
            if now - getattr(main, "_last_send", 0) >= FRAME_SEND_INTERVAL:
                main._last_send = now
                try:
                    b64 = encode_frame(frame)
                    resp = http.post("/frame", json={"image": b64})
                    last_card = resp.json()
                    card_type = last_card.get("type", "empty")
                    last_mode = {
                        "person": "mode1",
                        "recall": "mode2",
                        "unknown": "mode3",
                        "empty": "idle",
                    }.get(card_type, "idle")
                except Exception:
                    pass

            # ── Render overlay ──
            display = frame.copy()
            if last_card:
                t = last_card.get("type", "empty")
                if t == "person":
                    display = draw_person_card(display, last_card)
                elif t == "recall":
                    display = draw_recall_card(display, last_card)
                elif t == "unknown":
                    display = draw_unknown(display)

            display = draw_transcript_bar(display, audio_state)
            display = draw_status_bar(display, fps, audio_state)
            cv2.imshow("MemoryLens", display)

    except KeyboardInterrupt:
        pass
    finally:
        audio.stop()
        cap.release()
        cv2.destroyAllWindows()
        http.close()


def draw_unknown(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_UNKNOWN, 3)
    cv2.putText(frame, "Someone new", (20, 35),
                FONT, FONT_SCALE_NAME, COLOR_UNKNOWN, FONT_THICK)
    return frame


if __name__ == "__main__":
    main()
