from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import time


@dataclass(slots=True)
class AudioChunk:
    """In-memory PCM chunk produced by Streaming VAD."""

    chunk_id: str
    pcm16: bytes
    sample_rate: int
    # True if this chunk corresponds to an utterance end from VAD
    is_final: bool = False
    created_at: float = field(default_factory=time.time)

    def to_float32(self) -> np.ndarray:
        """Return normalised float32 waveform [-1, 1]."""
        return np.frombuffer(self.pcm16, dtype=np.int16).astype(np.float32) / 32768.0

    def __len__(self) -> int:
        return len(self.pcm16) // 2

    @property
    def duration_ms(self) -> float:
        return (len(self) / self.sample_rate) * 1000.0
