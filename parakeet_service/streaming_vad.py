from __future__ import annotations
import uuid
from typing import List

import numpy as np
from torch.hub import load as torch_hub_load

from .config import (
    STREAM_CHUNK_MS,
    STREAM_MAX_UNFLUSHED_MS,
    STREAM_RIGHT_CONTEXT_MS,
    SAMPLE_RATE,
    VAD_WINDOW_SAMPLES,
    VAD_THRESHOLD,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    logger,
)
from .types import AudioChunk

vad_model, vad_utils = torch_hub_load("snakers4/silero-vad", "silero_vad")  # type: ignore[misc]
(_, _, _, VADIterator, _) = vad_utils

# VAD / streaming constants are now configurable via `parakeet_service.config` (environment variables)
WINDOW_SAMPLES = VAD_WINDOW_SAMPLES
THRESHOLD = VAD_THRESHOLD
MIN_SILENCE_MS = VAD_MIN_SILENCE_MS
SPEECH_PAD_MS = VAD_SPEECH_PAD_MS

# Helper: float32 → int16 PCM bytes
def _f32_to_pcm16(frames: np.ndarray) -> bytes:
    return np.clip(frames * 32768, -32768, 32767).astype(np.int16).tobytes()

class StreamingVAD:
    """
    Feed successive 20–40 ms PCM frames (16 kHz, int16 mono).
    Emits in-memory ``AudioChunk`` objects when speech utterances complete.
    """

    def __init__(self):
        self.vad = VADIterator(
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=THRESHOLD,
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        )
        self.buffer = bytearray()
        # windows in current output chunk buffer (resets after target flush)
        self.active_windows = 0
        # total windows since current speech started (resets only on end/max flush)
        self._speech_windows_total = 0
        self.in_speech = False
        self._target_windows = max(1, STREAM_CHUNK_MS // 32)
        # Max total windows allowed between VAD 'start' and forced reset
        self._max_total_windows = max(self._target_windows, STREAM_MAX_UNFLUSHED_MS // 32)
        # Cap a single output chunk size by (chunk + right) to avoid oversized chunks
        self._max_chunk_windows = max(1, (STREAM_CHUNK_MS + STREAM_RIGHT_CONTEXT_MS) // 32)
        logger.debug(
            "VAD init: sr=%d, window=%d samp (%.1f ms), thresh=%.2f, min_sil=%d ms, pad=%d ms, target=%d win, max_total=%d win, max_chunk=%d win",
            SAMPLE_RATE,
            WINDOW_SAMPLES,
            1000.0 * WINDOW_SAMPLES / float(SAMPLE_RATE),
            THRESHOLD,
            MIN_SILENCE_MS,
            SPEECH_PAD_MS,
            self._target_windows,
            self._max_total_windows,
            self._max_chunk_windows,
        )

    def reset(self) -> None:
        self.buffer.clear()
        self.active_windows = 0
        self._speech_windows_total = 0
        self.in_speech = False
        self.vad.reset_states()
        logger.debug("VAD reset: buffers cleared")

    def _flush(self, reset_iterator: bool, *, final: bool) -> List[AudioChunk]:
        if not self.buffer:
            return []
        chunk = AudioChunk(
            chunk_id=uuid.uuid4().hex,
            pcm16=bytes(self.buffer),
            sample_rate=SAMPLE_RATE,
            is_final=final,
        )
        self.buffer.clear()
        self.active_windows = 0
        if reset_iterator:
            self.vad.reset_states()
        return [chunk]

    def feed(self, frame_bytes: bytes) -> List[AudioChunk]:
        out: List[AudioChunk] = []

        pcm_f32 = np.frombuffer(frame_bytes, np.int16).astype("float32") / 32768
        #logger.debug("VAD feed: got %d bytes (%.1f ms)", len(frame_bytes), 1000.0 * len(pcm_f32) / SAMPLE_RATE)
        pad_window_count = max(SPEECH_PAD_MS // 32, 0)
        lead_windows: List[np.ndarray] = []

        for start in range(0, len(pcm_f32), WINDOW_SAMPLES):
            window = pcm_f32[start:start + WINDOW_SAMPLES]
            if len(window) < WINDOW_SAMPLES:
                break  # wait for full 32 ms window

            voice_event = self.vad(window, return_seconds=False)
            if voice_event:
                logger.debug("VAD evt: %s", voice_event)

            # Maintain leading windows so we can prepend pad when speech begins
            lead_windows.append(window)
            if len(lead_windows) > pad_window_count:
                lead_windows.pop(0)

            buffered_now = False
            if voice_event and voice_event.get("start") is not None:
                # Entering speech: prepend lead windows and include the current window
                self.in_speech = True
                for buffered in lead_windows:
                    self.buffer.extend(_f32_to_pcm16(buffered))
                # append the current window that triggered the start
                self.buffer.extend(_f32_to_pcm16(window))
                # Initialize counters
                self.active_windows = len(lead_windows) + 1
                self._speech_windows_total = len(lead_windows) + 1
                lead_windows.clear()
                buffered_now = True
                logger.debug(
                    "VAD start: prime=%d win, active=%d, total=%d, buf_ms=%.1f",
                    pad_window_count,
                    self.active_windows,
                    self._speech_windows_total,
                    1000.0 * (len(self.buffer) // 2) / SAMPLE_RATE,
                )

            elif self.in_speech:
                # Normal speech continuation: append this window
                self.buffer.extend(_f32_to_pcm16(window))
                self.active_windows = min(self.active_windows + 1, self._max_chunk_windows)
                self._speech_windows_total += 1
                buffered_now = True
                logger.debug(
                    "VAD speech: +1 win -> active=%d, total=%d, buf_ms=%.1f",
                    self.active_windows,
                    self._speech_windows_total,
                    1000.0 * (len(self.buffer) // 2) / SAMPLE_RATE,
                )

            if not buffered_now:
                # still waiting for speech; do not emit chunks yet
                continue

            hit_target = self.active_windows >= self._target_windows
            hit_max_total = self._speech_windows_total >= self._max_total_windows
            ended = bool(voice_event and voice_event.get("end"))

            if ended:
                out.extend(self._flush(reset_iterator=True, final=True))
                self.in_speech = False
                self._speech_windows_total = 0
                lead_windows.clear()
                logger.debug("VAD end: flush(end) -> chunks=%d", len(out))
            elif hit_target:
                # Mid-speech partial flush; keep VAD states so we continue appending
                out.extend(self._flush(reset_iterator=False, final=False))
                logger.debug("VAD flush: hit target (%d win)", self._target_windows)
            elif hit_max_total:
                out.extend(self._flush(reset_iterator=True, final=True))
                self.in_speech = False
                self._speech_windows_total = 0
                lead_windows.clear()
                logger.debug("VAD flush: hit max_total (%d win), reset VAD", self._max_total_windows)

        return out
