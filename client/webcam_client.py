"""
MemoryLens Webcam Overlay Client

Replaces the OLED/glasses display. Captures webcam frames, sends them
to the FastAPI server, and renders HUD cards as OpenCV overlays.

Usage:
    python -m client.webcam_client

Controls:
    e  — enroll the currently visible person
    r  — trigger recall (enter a confusion phrase)
    q  — quit
"""

import base64
import json
import sys
import time
import threading
from pathlib import Path

import cv2
import httpx
import numpy as np

# ── Config ────────────────────────────────────────────────────────────

SERVER_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
WEBCAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
SEND_INTERVAL = 0.33  # ~3 FPS to server

# ── Colors (BGR) ─────────────────────────────────────────────────────

COLOR_KNOWN = (0, 200, 0)       # green border — confirmed person
COLOR_AUTO = (0, 180, 255)      # orange border — auto-enrolled
COLOR_RECALL = (255, 100, 0)    # blue border — recall narration
COLOR_UNKNOWN = (0, 0, 255)     # red border — unknown face
COLOR_EMPTY = (128, 128, 128)   # grey — no face detected
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (30, 30, 30)

# ── Font ──────────────────────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_NAME = 0.7
FONT_SCALE = 0.45
FONT_THICK = 2
FONT_THIN = 1


# ── Helpers ───────────────────────────────────────────────────────────

def encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("utf-8")


def wrap_text(text: str, max_width: int, font, scale, thickness) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    line = ""
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
    """Render a person card overlay on the frame."""
    h, w = frame.shape[:2]
    status = card.get("enrollment_status", "confirmed")
    border_color = COLOR_KNOWN if status == "confirmed" else COLOR_AUTO

    # Border
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 3)

    # Card background (top-left)
    card_w, card_h = 320, 160
    overlay = frame[10:10 + card_h, 10:10 + card_w].copy()
    cv2.rectangle(frame, (10, 10), (10 + card_w, 10 + card_h), COLOR_BG, -1)

    y = 30
    # Name
    cv2.putText(frame, card.get("name", "?"), (20, y),
                FONT, FONT_SCALE_NAME, COLOR_TEXT, FONT_THICK)
    y += 25

    # Relation + last seen
    relation_line = f"{card.get('relation', '')}  |  {card.get('last_seen', '')}"
    cv2.putText(frame, relation_line, (20, y),
                FONT, FONT_SCALE, (200, 200, 200), FONT_THIN)
    y += 5

    # Divider
    cv2.line(frame, (20, y), (card_w, y), (80, 80, 80), 1)
    y += 18

    # Summary (word-wrapped)
    summary = card.get("summary", "")
    lines = wrap_text(summary, card_w - 20, FONT, FONT_SCALE, FONT_THIN)
    for line in lines[:4]:  # max 4 lines
        cv2.putText(frame, line, (20, y),
                    FONT, FONT_SCALE, (180, 180, 180), FONT_THIN)
        y += 18

    return frame


def draw_recall_card(frame: np.ndarray, card: dict) -> np.ndarray:
    """Render a recall narration overlay — full-width at bottom."""
    h, w = frame.shape[:2]
    narration = card.get("narration", "")

    # Bottom bar
    bar_h = 120
    cv2.rectangle(frame, (0, h - bar_h), (w, h), COLOR_BG, -1)
    cv2.rectangle(frame, (0, h - bar_h), (w, h), COLOR_RECALL, 2)

    # Label
    cv2.putText(frame, "RECALL", (15, h - bar_h + 22),
                FONT, 0.5, COLOR_RECALL, FONT_THIN)

    # Narration text
    lines = wrap_text(narration, w - 30, FONT, FONT_SCALE, FONT_THIN)
    y = h - bar_h + 42
    for line in lines[:3]:
        cv2.putText(frame, line, (15, y),
                    FONT, FONT_SCALE, COLOR_TEXT, FONT_THIN)
        y += 20

    return frame


def draw_unknown_card(frame: np.ndarray, card: dict) -> np.ndarray:
    """Red border + 'Someone new' label."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_UNKNOWN, 3)
    cv2.putText(frame, card.get("label", "Someone new"), (20, 35),
                FONT, FONT_SCALE_NAME, COLOR_UNKNOWN, FONT_THICK)
    return frame


def draw_status_bar(frame: np.ndarray, fps: float, last_mode: str) -> np.ndarray:
    """Top-right status indicators."""
    h, w = frame.shape[:2]
    status_text = f"FPS: {fps:.0f}  |  Mode: {last_mode}"
    (tw, _), _ = cv2.getTextSize(status_text, FONT, 0.4, 1)
    cv2.putText(frame, status_text, (w - tw - 15, 25),
                FONT, 0.4, (100, 100, 100), 1)

    # Controls hint
    hint = "e=enroll  r=recall  q=quit"
    (hw, _), _ = cv2.getTextSize(hint, FONT, 0.35, 1)
    cv2.putText(frame, hint, (w - hw - 15, 45),
                FONT, 0.35, (80, 80, 80), 1)
    return frame


# ── Enrollment dialog ─────────────────────────────────────────────────

def enrollment_dialog(frame: np.ndarray) -> dict | None:
    """
    Simple text input using OpenCV highgui.
    Returns {name, relation} or None if cancelled.
    """
    name = ""
    relation = ""
    step = "name"

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

        if key == 27:  # ESC — cancel
            return None
        elif key == 13:  # Enter
            if step == "name" and name:
                step = "relation"
            elif step == "relation":
                return {"name": name, "relation": relation or "unknown"}
        elif key == 8:  # Backspace
            if step == "name":
                name = name[:-1]
            else:
                relation = relation[:-1]
        elif 32 <= key <= 126:
            char = chr(key)
            if step == "name":
                name += char
            else:
                relation += char


def recall_dialog(frame: np.ndarray) -> str | None:
    """Type a confusion phrase. Returns the phrase or None if cancelled."""
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


# ── Main loop ─────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open webcam {WEBCAM_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    http = httpx.Client(base_url=SERVER_URL, timeout=5.0)
    last_card = None
    last_mode = "idle"
    frame_count = 0
    fps_timer = time.time()
    fps = 0.0

    print(f"MemoryLens client connected to {SERVER_URL}")
    print("Controls: e=enroll  r=recall  q=quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS counter
            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = frame_count / (time.time() - fps_timer)
                frame_count = 0
                fps_timer = time.time()

            # Check for keypress (non-blocking after first frame)
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
                            "session_context": "",
                        })
                        last_card = resp.json()
                        last_mode = "recall"
                    except Exception as e:
                        print(f"Recall failed: {e}")
                    continue

            # Send frame to server (throttled)
            now = time.time()
            if now - getattr(main, "_last_send", 0) >= SEND_INTERVAL:
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
                    pass  # Server unreachable — keep last card

            # Render overlay
            display = frame.copy()
            if last_card:
                t = last_card.get("type", "empty")
                if t == "person":
                    display = draw_person_card(display, last_card)
                elif t == "recall":
                    display = draw_recall_card(display, last_card)
                elif t == "unknown":
                    display = draw_unknown_card(display, last_card)

            display = draw_status_bar(display, fps, last_mode)
            cv2.imshow("MemoryLens", display)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        http.close()


if __name__ == "__main__":
    main()
