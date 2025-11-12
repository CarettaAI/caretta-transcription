from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import threading

import torch
from nemo.collections.asr.parts.submodules.transducer_decoding.label_looping_base import (
    BatchedLabelLoopingState,
    GreedyBatchedLabelLoopingComputerBase,
)
from nemo.collections.asr.parts.utils.rnnt_utils import BatchedHyps, batched_hyps_to_hypotheses
from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer

from .config import (
    STREAM_CHUNK_SECS,
    STREAM_LEFT_CONTEXT_SECS,
    STREAM_RIGHT_CONTEXT_SECS,
    TARGET_SR,
    logger,
)
from .types import AudioChunk


def _make_divisible_by(num: int, factor: int) -> int:
    if factor <= 0:
        return num
    if num <= 0:
        return factor
    return (num // factor) * factor


@dataclass(slots=True)
class StreamTask:
    conn_id: str
    chunk: AudioChunk


@dataclass(slots=True)
class StreamResult:
    conn_id: str
    chunk_id: str
    text: str
    delta: str
    is_final: bool = False


class StreamingSession:
    """Holds decoder state for a single websocket connection."""

    __slots__ = (
        "conn_id",
        "engine",
        "buffer",
        "state",
        "hyps",
        "last_text",
        "is_closed",
    )

    def __init__(self, conn_id: str, engine: "StreamingEngine") -> None:
        self.conn_id = conn_id
        self.engine = engine
        self.buffer = StreamingBatchedAudioBuffer(
            batch_size=1,
            context_samples=engine.context_samples,
            dtype=engine.buffer_dtype,
            device=engine.device,
        )
        self.state: Optional[BatchedLabelLoopingState] = None
        self.hyps: Optional[BatchedHyps] = None
        self.last_text: str = ""
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True
        self.state = None
        self.hyps = None
        # let GPU memory be reclaimed lazily; tensors will be GC'd

    def consume_chunk(self, chunk: AudioChunk) -> Optional[StreamResult]:
        if self.is_closed:
            logger.debug("Skipping chunk for closed session %s", self.conn_id)
            return None
        if chunk.sample_rate != TARGET_SR:
            raise ValueError(
                f"Unsupported sample rate {chunk.sample_rate}; expected {TARGET_SR} Hz"
            )

        device = self.engine.device
        audio = torch.from_numpy(chunk.to_float32()).to(device=device)
        if audio.ndim != 1:
            audio = audio.squeeze(0)
        audio = audio.unsqueeze(0).to(dtype=self.engine.buffer_dtype)
        frame_count = audio.shape[1]
        if frame_count == 0:
            return None

        is_last = torch.tensor([False], dtype=torch.bool, device=device)
        audio_lengths = torch.tensor([frame_count], dtype=torch.long, device=device)
        self.buffer.add_audio_batch_(audio, audio_lengths, False, is_last)

        text = self._decode_and_update()
        if text is None:
            return None

        if text.startswith(self.last_text):
            delta = text[len(self.last_text) :]
        else:
            delta = text
        self.last_text = text

        return StreamResult(
            conn_id=self.conn_id,
            chunk_id=chunk.chunk_id,
            text=text,
            delta=delta,
            is_final=False,
        )

    def _decode_and_update(self) -> Optional[str]:
        engine = self.engine
        buffer = self.buffer

        input_signal = buffer.samples
        if input_signal.numel() == 0:
            return None
        input_lengths = buffer.context_size_batch.total()
        encoder_output, _ = engine.model(
            input_signal=input_signal,
            input_signal_length=input_lengths,
        )
        encoder_output = encoder_output.transpose(1, 2)

        encoder_context = buffer.context_size.subsample(engine.encoder_frame2audio_samples)
        encoder_context_batch = buffer.context_size_batch.subsample(engine.encoder_frame2audio_samples)
        encoder_output = encoder_output[:, encoder_context.left :]

        chunk_hyps, _, decoder_state = engine.decoding_computer(
            x=encoder_output,
            out_len=encoder_context_batch.chunk,
            prev_batched_state=self.state,
        )
        self.state = decoder_state
        if self.hyps is None:
            self.hyps = chunk_hyps
        else:
            self.hyps.merge_(chunk_hyps)

        hypotheses = batched_hyps_to_hypotheses(self.hyps, batch_size=1)
        if not hypotheses:
            return None
        sequence = hypotheses[0].y_sequence
        if isinstance(sequence, torch.Tensor):
            tokens = sequence.cpu().tolist()
        else:
            tokens = list(sequence)
        return engine.model.tokenizer.ids_to_text(tokens)


class StreamingEngine:
    """Coordinates streaming ASR across multiple websocket connections."""

    def __init__(self, model) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.buffer_dtype = torch.float32
        self.decoding_computer: GreedyBatchedLabelLoopingComputerBase = (
            model.decoding.decoding.decoding_computer
        )

        sample_rate = int(model.cfg.preprocessor["sample_rate"])
        self.sample_rate = sample_rate
        feature_stride_sec = model.cfg.preprocessor["window_stride"]
        features_per_sec = 1.0 / feature_stride_sec
        subsampling_factor = model.encoder.subsampling_factor

        features_frame2audio_samples = _make_divisible_by(
            int(sample_rate * feature_stride_sec), factor=subsampling_factor
        )
        self.encoder_frame2audio_samples = features_frame2audio_samples * subsampling_factor

        def _frames(sec: float) -> int:
            frames = int(sec * features_per_sec / subsampling_factor)
            return max(frames, 1)

        self.context_encoder_frames = ContextSize(
            left=_frames(STREAM_LEFT_CONTEXT_SECS),
            chunk=_frames(STREAM_CHUNK_SECS),
            right=_frames(STREAM_RIGHT_CONTEXT_SECS),
        )
        self.context_samples = ContextSize(
            left=self.context_encoder_frames.left
            * subsampling_factor
            * features_frame2audio_samples,
            chunk=self.context_encoder_frames.chunk
            * subsampling_factor
            * features_frame2audio_samples,
            right=self.context_encoder_frames.right
            * subsampling_factor
            * features_frame2audio_samples,
        )

        self._sessions: Dict[str, StreamingSession] = {}
        self._lock = threading.Lock()

        logger.info(
            "Streaming engine initialised (context frames L=%d C=%d R=%d, samples=%d)",
            self.context_encoder_frames.left,
            self.context_encoder_frames.chunk,
            self.context_encoder_frames.right,
            self.context_samples.total(),
        )

    def create_session(self, conn_id: str) -> StreamingSession:
        session = StreamingSession(conn_id, self)
        with self._lock:
            self._sessions[conn_id] = session
        return session

    def close_session(self, conn_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(conn_id, None)
        if session:
            session.close()

    def process_batch(self, tasks: List[StreamTask]) -> List[StreamResult]:
        if not tasks:
            return []

        with self._lock:
            work = [
                (self._sessions.get(task.conn_id), task)
                for task in tasks
            ]

        results: List[StreamResult] = []
        with torch.inference_mode():
            for session, task in work:
                if session is None:
                    logger.debug(
                        "Dropping chunk %s for missing session %s",
                        task.chunk.chunk_id,
                        task.conn_id,
                    )
                    continue
                try:
                    result = session.consume_chunk(task.chunk)
                except Exception:  # pragma: no cover - defensive log
                    logger.exception(
                        "Streaming decode failed for session %s", task.conn_id
                    )
                    continue
                if result:
                    results.append(result)
        return results

    def active_session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()
