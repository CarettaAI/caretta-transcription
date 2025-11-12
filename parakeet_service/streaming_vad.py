from __future__ import annotations
import uuid
from typing import List

import numpy as np
from torch.hub import load as torch_hub_load

from .config import (
    STREAM_CHUNK_MS,
    STREAM_MAX_UNFLUSHED_MS,
    SAMPLE_RATE,
    VAD_WINDOW_SAMPLES,
    VAD_THRESHOLD,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
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
        self.active_windows = 0
        self.in_speech = False
        self._target_windows = max(1, STREAM_CHUNK_MS // 32)
        self._max_windows = max(self._target_windows, STREAM_MAX_UNFLUSHED_MS // 32)

    def reset(self) -> None:
        self.buffer.clear()
        self.active_windows = 0
        self.in_speech = False
        self.vad.reset_states()

    def _flush(self, reset_iterator: bool) -> List[AudioChunk]:
        if not self.buffer:
            return []
        chunk = AudioChunk(
            chunk_id=uuid.uuid4().hex,
            pcm16=bytes(self.buffer),
            sample_rate=SAMPLE_RATE,
        )
        self.buffer.clear()
        self.active_windows = 0
        if reset_iterator:
            self.vad.reset_states()
        return [chunk]

    def feed(self, frame_bytes: bytes) -> List[AudioChunk]:
        out: List[AudioChunk] = []

        pcm_f32 = np.frombuffer(frame_bytes, np.int16).astype("float32") / 32768
        for start in range(0, len(pcm_f32), WINDOW_SAMPLES):
            window = pcm_f32[start:start + WINDOW_SAMPLES]
            if len(window) < WINDOW_SAMPLES:
                break  # wait for full 32 ms window

            voice_event = self.vad(window, return_seconds=False)
            self.buffer.extend(_f32_to_pcm16(window))
            self.active_windows += 1

            if voice_event and voice_event.get("start") is not None:
                self.in_speech = True

            hit_target = self.in_speech and self.active_windows >= self._target_windows
            hit_max = self.active_windows >= self._max_windows
            ended = bool(voice_event and voice_event.get("end"))

            if ended:
                out.extend(self._flush(reset_iterator=True))
                self.in_speech = False
            elif hit_target:
                out.extend(self._flush(reset_iterator=False))
            elif hit_max:
                out.extend(self._flush(reset_iterator=True))
                self.in_speech = False

        return out
