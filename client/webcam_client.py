"""
MemoryLens Webcam Overlay Client (WebSocket)

Captures webcam frames + mic audio, streams to the FastAPI server via WebSocket,
and renders HUD cards as OpenCV overlays. Fully automatic.

Usage:
    python -m client.webcam_client [server_url]

Debug overrides:
    e  — manually enroll the currently visible person
    r  — manually trigger recall with typed phrase
    q  — quit
"""

import base64
import sys
import time
import threading
import queue
import asyncio
import json

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────

SERVER_URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8000/ws"
WEBCAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
FRAME_SEND_INTERVAL = 0.5   # send frame every 0.5s (server pushes updates)

# Audio
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SECONDS = 10
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
FONT_SCALE = 0.5
FONT_SCALE_NAME = 0.7
FONT_THICK = 2
FONT_THIN = 1


# ── Audio Capture ─────────────────────────────────────────────────────

class AudioCapture:
    """Captures mic audio into chunks for WebSocket streaming."""

    def __init__(self):
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_chunk(self) -> np.ndarray | None:
        try:
            return self._audio_queue.get_nowait()
        except queue.Empty:
            return None

    def _capture_loop(self):
        try:
            import sounddevice as sd
        except ImportError:
            print("WARNING: sounddevice not installed. Audio capture disabled.")
            return

        chunk_samples = AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SECONDS
        buffer = np.zeros(chunk_samples, dtype=np.int16)
        write_pos = 0

        def callback(indata, frames, time_info, status):
            nonlocal buffer, write_pos
            chunk = indata[:, 0].copy()
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
                blocksize=int(AUDIO_SAMPLE_RATE * 0.1),
                callback=callback,
            ):
                while self._running:
                    if write_pos >= chunk_samples:
                        self._audio_queue.put(buffer.copy())
                        write_pos = 0
        except Exception as e:
            print(f"Audio capture error: {e}")


# ── WebSocket Thread ──────────────────────────────────────────────────

class WebSocketClient:
    """Manages WebSocket connection to server in a background thread."""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self._send_queue: queue.Queue[dict] = queue.Queue()
        self._recv_queue: queue.Queue[dict] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self._connected = threading.Event()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def send(self, msg: dict):
        self._send_queue.put(msg)

    def recv(self) -> dict | None:
        try:
            return self._recv_queue.get_nowait()
        except queue.Empty:
            return None

    def _ws_loop(self):
        import websockets.sync.client as wsc

        while self._running:
            try:
                with wsc.connect(self.server_url) as ws:
                    self._connected.set()
                    print(f"WebSocket connected to {self.server_url}")

                    # Sender thread
                    def _sender():
                        while self._running:
                            try:
                                msg = self._send_queue.get(timeout=0.1)
                                ws.send(json.dumps(msg))
                            except queue.Empty:
                                continue
                            except Exception:
                                break

                    sender = threading.Thread(target=_sender, daemon=True)
                    sender.start()

                    # Receiver loop
                    while self._running:
                        try:
                            raw = ws.recv(timeout=1.0)
                            data = json.loads(raw)
                            self._recv_queue.put(data)
                        except TimeoutError:
                            continue
                        except Exception:
                            break

                    self._connected.clear()
            except Exception as e:
                if self._running:
                    self._connected.clear()
                    print(f"WebSocket disconnected, reconnecting in 3s... ({e})")
                    time.sleep(3)


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
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_RECALL, 3)

    card_w = 400
    cv2.rectangle(frame, (10, 10), (10 + card_w, 120), COLOR_BG, -1)
    y = 35
    cv2.putText(frame, "RECALL", (20, y), FONT, 0.6, COLOR_RECALL, FONT_THICK)
    y += 25
    narration = card.get("narration", "")
    for line in wrap_text(narration, card_w - 10, FONT, FONT_SCALE, FONT_THIN)[:4]:
        cv2.putText(frame, line, (20, y), FONT, FONT_SCALE, COLOR_TEXT, FONT_THIN)
        y += 18
    return frame


def draw_transcript_bar(frame: np.ndarray, transcript: str) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_y = h - 60
    cv2.rectangle(frame, (0, bar_y), (w, h), COLOR_BG, -1)
    cv2.putText(frame, "LIVE", (10, bar_y + 20),
                FONT, 0.4, (0, 200, 0), FONT_THIN)
    if transcript:
        text = transcript[:80]
        cv2.putText(frame, text, (10, bar_y + 42),
                    FONT, FONT_SCALE, COLOR_TRANSCRIPT, FONT_THIN)
    return frame


def draw_status_bar(frame: np.ndarray, fps: float, connected: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    status = f"FPS: {fps:.0f} | WS: {'ON' if connected else 'OFF'}"
    cv2.putText(frame, status, (w - 200, 20),
                FONT, 0.4, COLOR_TRANSCRIPT, FONT_THIN)
    cv2.putText(frame, "e=enroll r=recall q=quit", (w - 200, 38),
                FONT, 0.35, (120, 120, 120), FONT_THIN)
    return frame


def enrollment_dialog(frame: np.ndarray) -> dict | None:
    """Debug: type name + relation for manual enrollment."""
    text = ""
    while True:
        display = frame.copy()
        h, w = display.shape[:2]
        cv2.rectangle(display, (50, 150), (w - 50, 260), COLOR_BG, -1)
        cv2.rectangle(display, (50, 150), (w - 50, 260), COLOR_AUTO, 2)
        cv2.putText(display, "ENROLL", (70, 185),
                    FONT, 0.6, COLOR_AUTO, FONT_THICK)
        cv2.putText(display, f'"{text}_"', (70, 220),
                    FONT, 0.5, COLOR_TEXT, FONT_THIN)
        cv2.imshow("MemoryLens", display)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            return None
        elif key == 13 and text:
            parts = text.split(",", 1)
            name = parts[0].strip()
            relation = parts[1].strip() if len(parts) > 1 else "unknown"
            return {"name": name, "relation": relation}
        elif key == 8:
            text = text[:-1]
        elif 32 <= key <= 126:
            text += chr(key)


def recall_dialog(frame: np.ndarray) -> str | None:
    """Debug: type a phrase for manual recall."""
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

    ws_client = WebSocketClient(SERVER_URL)
    ws_client.start()

    audio = AudioCapture()
    audio.start()

    last_card = {"type": "empty"}
    frame_count = 0
    fps_timer = time.time()
    fps = 0.0
    last_frame_time = 0.0
    last_transcript = ""

    print(f"MemoryLens connecting to {SERVER_URL}")
    print("System is fully automatic. Debug: e=enroll  r=recall  q=quit")

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

            # Send frame to server (throttled)
            now = time.time()
            if ws_client.is_connected() and (now - last_frame_time) >= FRAME_SEND_INTERVAL:
                last_frame_time = now
                b64 = encode_frame(frame)
                ws_client.send({"type": "frame", "image": b64})

            # Send audio chunks
            chunk = audio.get_chunk()
            if chunk is not None and ws_client.is_connected():
                b64 = base64.b64encode(chunk.tobytes()).decode()
                ws_client.send({"type": "audio", "audio": b64, "sample_rate": AUDIO_SAMPLE_RATE})

            # Receive server updates
            while True:
                msg = ws_client.recv()
                if msg is None:
                    break
                msg_type = msg.get("type", "")
                if msg_type in ("person", "unknown", "recall", "empty"):
                    last_card = msg
                elif msg_type == "transcript":
                    last_transcript = msg.get("full_text", "")
                elif msg_type == "error":
                    print(f"Server error: {msg.get('error')}")

            # Debug overrides
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("e"):
                result = enrollment_dialog(frame)
                if result and ws_client.is_connected():
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    b64 = base64.b64encode(buf).decode()
                    ws_client.send({
                        "type": "enroll",
                        "name": result["name"],
                        "relation": result["relation"],
                        "image": b64,
                    })
                    print(f"Enrolled: {result['name']} ({result['relation']})")
                continue
            elif key == ord("r"):
                phrase = recall_dialog(frame)
                if phrase and ws_client.is_connected():
                    person_id = last_card.get("person_id") if last_card.get("type") == "person" else None
                    ws_client.send({
                        "type": "recall",
                        "trigger_phrase": phrase,
                        "confirmed_person_id": person_id,
                        "session_context": last_transcript,
                    })
                continue

            # Render overlay
            display = frame.copy()
            t = last_card.get("type", "empty")
            if t == "person":
                display = draw_person_card(display, last_card)
            elif t == "recall":
                display = draw_recall_card(display, last_card)
            elif t == "unknown":
                display = draw_unknown(display)

            display = draw_transcript_bar(display, last_transcript)
            display = draw_status_bar(display, fps, ws_client.is_connected())
            cv2.imshow("MemoryLens", display)

    finally:
        audio.stop()
        ws_client.stop()
        cap.release()
        cv2.destroyAllWindows()


def draw_unknown(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_UNKNOWN, 3)
    cv2.putText(frame, "Someone new", (20, 35),
                FONT, FONT_SCALE_NAME, COLOR_UNKNOWN, FONT_THICK)
    return frame


if __name__ == "__main__":
    main()
