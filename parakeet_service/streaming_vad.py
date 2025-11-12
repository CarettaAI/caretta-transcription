from __future__ import annotations
import uuid
import numpy as np, torch
from typing import List
from torch.hub import load as torch_hub_load

from parakeet_service.types import AudioChunk

vad_model, vad_utils = torch_hub_load("snakers4/silero-vad", "silero_vad")  # type: ignore[misc]
(_, _, _, VADIterator, _) = vad_utils

# TODO: Update to read from .env
SAMPLE_RATE              = 16_000         # model is trained for 16 kHz
WINDOW_SAMPLES           = 512            # 32 ms frame
THRESHOLD                = 0.60           # voice prob ≥ 0.60 → speech
MIN_SILENCE_MS           = 250            # flush after ≥250 ms quiet
SPEECH_PAD_MS            = 120            # keep 120 ms context before/after
MAX_SPEECH_MS            = 8_000          # hard stop at 8 s

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
        self.speech_ms = 0


    def _flush(self) -> List[AudioChunk]:
        if not self.buffer:
            return []
        chunk = AudioChunk(
            chunk_id=uuid.uuid4().hex,
            pcm16=bytes(self.buffer),
            sample_rate=SAMPLE_RATE,
        )
        self.buffer.clear()
        self.speech_ms = 0
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
            self.speech_ms += 32

            # Flush on trailing-silence event or max-length guard
            if voice_event and voice_event.get("end"):
                out.extend(self._flush())
            elif self.speech_ms >= MAX_SPEECH_MS:
                out.extend(self._flush())

        return out
