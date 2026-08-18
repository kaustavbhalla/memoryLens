"""Audio pipeline — WhisperX for STT + diarization, speaker embeddings for identity."""

from __future__ import annotations

import logging
import tempfile
import os
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from server.config import (
    COMPUTE_DEVICE,
    COMPUTE_TYPE,
    WHISPERX_MODEL,
    WHISPERX_BATCH_SIZE,
    WHISPERX_LANGUAGE,
    WHISPERX_MIN_SPEAKERS,
    WHISPERX_MAX_SPEAKERS,
    HUGGINGFACE_TOKEN,
    AUDIO_CHUNK_DURATION_S,
)

log = logging.getLogger("memorylens.audio")


@dataclass
class SpeakerSegment:
    speaker: str        # "SPEAKER_00", "SPEAKER_01", ...
    start: float        # seconds
    end: float          # seconds
    text: str           # transcribed text for this segment
    embedding: np.ndarray | None = None  # 256-dim voice embedding


@dataclass
class AudioResult:
    segments: list[SpeakerSegment] = field(default_factory=list)
    full_text: str = ""
    language: str = ""

    @property
    def speaker_texts(self) -> dict[str, str]:
        """Merge all segments per speaker into a single string."""
        result: dict[str, str] = {}
        for seg in self.segments:
            if seg.speaker not in result:
                result[seg.speaker] = ""
            result[seg.speaker] += " " + seg.text
        return {k: v.strip() for k, v in result.items()}

    @property
    def speaker_turns(self) -> list[dict]:
        """Chronological speaker turns as dicts."""
        return [
            {"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text}
            for s in self.segments
        ]


class AudioPipeline:
    """
    Unified audio processing using WhisperX:
    1. Transcribe (faster-whisper backend, batched)
    2. Align (wav2vec2 forced alignment → word-level timestamps)
    3. Diarize (pyannote speaker segmentation)
    4. Assign speakers to words/segments

    Also extracts per-turn speaker embeddings for voice identity matching.
    """

    def __init__(self):
        self._whisperx_model = None
        self._align_model = None
        self._align_metadata = None
        self._diarize_model = None
        self._loaded = False

    async def load(self):
        """Load all WhisperX models. Call at startup."""
        import whisperx

        log.info(f"Loading WhisperX model '{WHISPERX_MODEL}' on {COMPUTE_DEVICE} ({COMPUTE_TYPE})...")

        # 1. Transcription model (faster-whisper backend)
        self._whisperx_model = whisperx.load_model(
            WHISPERX_MODEL,
            COMPUTE_DEVICE,
            compute_type=COMPUTE_TYPE,
            language=WHISPERX_LANGUAGE,
        )

        # 2. Alignment model (wav2vec2)
        log.info("Loading alignment model...")
        lang = WHISPERX_LANGUAGE or "en"
        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=lang,
            device=COMPUTE_DEVICE,
        )

        # 3. Diarization model (pyannote)
        if HUGGINGFACE_TOKEN:
            log.info("Loading diarization model...")
            from whisperx.diarize import DiarizationPipeline
            self._diarize_model = DiarizationPipeline(
                token=HUGGINGFACE_TOKEN,
                device=COMPUTE_DEVICE,
            )
        else:
            log.warning("No HF_TOKEN — diarization disabled. Speaker labels will not be assigned.")

        self._loaded = True
        log.info("Audio pipeline ready")

    def process_chunk(self, audio: np.ndarray, sample_rate: int = 16000) -> AudioResult:
        """
        Process a 30s audio chunk through the full WhisperX pipeline.
        audio: numpy array, float32 or int16, mono.
        Returns AudioResult with diarized, speaker-assigned segments.
        """
        if not self._loaded:
            raise RuntimeError("AudioPipeline not loaded — call load() first")

        import whisperx

        # Ensure float32 mono
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Write to temp WAV for WhisperX
        tmp_path = tempfile.mktemp(suffix=".wav")
        try:
            sf.write(tmp_path, audio, sample_rate)

            # Step 1: Transcribe
            result = self._whisperx_model.transcribe(
                audio,
                batch_size=WHISPERX_BATCH_SIZE,
                language=WHISPERX_LANGUAGE,
            )

            # Step 2: Align (word-level timestamps)
            result = whisperx.align(
                result["segments"],
                self._align_model,
                self._align_metadata,
                audio,
                COMPUTE_DEVICE,
                return_char_alignments=False,
            )

            # Step 3: Diarize
            if self._diarize_model is not None:
                diarize_segments = self._diarize_model(
                    audio,
                    min_speakers=WHISPERX_MIN_SPEAKERS,
                    max_speakers=WHISPERX_MAX_SPEAKERS,
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)

            # Step 4: Extract speaker embeddings from diarized segments
            segments = []
            full_text_parts = []

            for seg in result.get("segments", []):
                speaker = seg.get("speaker", "UNKNOWN")
                text = seg.get("text", "").strip()
                if not text:
                    continue

                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)

                # Extract audio for this segment to get speaker embedding
                embedding = self._extract_speaker_embedding(audio, start, end, sample_rate)

                segments.append(SpeakerSegment(
                    speaker=speaker,
                    start=start,
                    end=end,
                    text=text,
                    embedding=embedding,
                ))
                full_text_parts.append(text)

            return AudioResult(
                segments=segments,
                full_text=" ".join(full_text_parts),
                language=result.get("language", WHISPERX_LANGUAGE or "en"),
            )

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _extract_speaker_embedding(
        self, audio: np.ndarray, start: float, end: float, sample_rate: int
    ) -> np.ndarray | None:
        """Extract a speaker embedding from a diarized segment."""
        try:
            # Need at least 1 second of audio for a meaningful embedding
            if (end - start) < 0.5:
                return None

            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_audio = audio[start_sample:end_sample]

            # Cache the embedder — creating it every call is slow
            if not hasattr(self, "_spk_embedder") or self._spk_embedder is None:
                from speechbrain.inference import EncoderClassifier
                import torch
                self._spk_embedder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": COMPUTE_DEVICE},
                )

            import torch
            waveform = torch.tensor(segment_audio).unsqueeze(0).float()
            if COMPUTE_DEVICE == "cuda":
                waveform = waveform.cuda()
            with torch.no_grad():
                embedding = self._spk_embedder.encode_batch(waveform)
            return embedding.squeeze().cpu().numpy()

        except Exception as e:
            log.warning(f"Failed to extract speaker embedding: {e}")
            return None
