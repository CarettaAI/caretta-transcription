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
        "_pending",
        "_primed",
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
        # Pending audio samples not yet fed to buffer; we will feed only exact
        # (chunk+right) sized blocks and aligned to encoder frames. Shape: [1, n]
        self._pending: Optional[torch.Tensor] = None
        # Whether the buffer has received the initial (chunk+right) priming block
        self._primed: bool = False

    def close(self) -> None:
        self.is_closed = True
        self.state = None
        self.hyps = None
        self._pending = None
        self._primed = False
        # let GPU memory be reclaimed lazily; tensors will be GC'd
    
    def _reset_for_new_utterance(self) -> None:
        """Reset decoder state and recreate buffer for next utterance."""
        self.state = None
        self.hyps = None
        self._pending = None
        self._primed = False
        # Recreate buffer to reset NeMo's internal context tracking
        self.buffer = StreamingBatchedAudioBuffer(
            batch_size=1,
            context_samples=self.engine.context_samples,
            dtype=self.engine.buffer_dtype,
            device=self.engine.device,
        )
        self.last_text = ""
        logger.debug("[%s] reset for new utterance", self.conn_id)

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
        logger.debug(
            "[%s] consume_chunk: got chunk_id=%s, len=%d samp (%.1f ms), pending=%d, primed=%s",
            self.conn_id,
            chunk.chunk_id,
            audio.shape[1],
            1000.0 * audio.shape[1] / float(TARGET_SR),
            0 if (self._pending is None) else int(self._pending.shape[1]),
            self._primed,
        )

        # Append to pending buffer
        if self._pending is not None and self._pending.numel() > 0:
            audio = torch.cat([self._pending, audio], dim=1)
            logger.debug("[%s] pending concat -> total=%d samp", self.conn_id, int(audio.shape[1]))

        if audio.shape[1] == 0:
            return None

        frame_samples = self.engine.encoder_frame2audio_samples
        # We can only advance in full encoder frames
        total_full = (audio.shape[1] // frame_samples) * frame_samples
        if total_full == 0:
            self._pending = audio  # wait for more samples
            logger.debug("[%s] not frame-aligned yet -> stash=%d", self.conn_id, int(audio.shape[1]))
            return None

        # Determine how much to feed: we always feed fixed chunk-sized blocks except for the
        # final chunk which may be shorter.
        chunk_samples = self.engine.context_samples.chunk

        # Keep only fully frame-aligned portion for processing; stash tail
        processable = audio[:, :total_full]
        tail = audio[:, total_full:]
        if chunk.is_final and tail.numel() > 0:
            logger.debug("[%s] final chunk discarding %d non-frame samples", self.conn_id, int(tail.shape[1]))
            tail = tail[:, 0:0]
        if tail.numel() > 0:
            logger.debug("[%s] tail not full frame: %d samp stashed", self.conn_id, int(tail.shape[1]))

        latest_text: Optional[str] = None
        # Helper to feed the buffer with proper last-chunk flags
        def _feed_buffer(block: torch.Tensor, is_last_chunk: bool) -> Optional[str]:
            if block.shape[1] == 0:
                return None
            audio_lengths = torch.tensor([block.shape[1]], dtype=torch.long, device=device)
            last_flag = torch.tensor([is_last_chunk], dtype=torch.bool, device=device)
            self.buffer.add_audio_batch_(block, audio_lengths, is_last_chunk, last_flag)
            return self._decode_and_update()

        # Feed as many full blocks as available
        
        # Strategy: Always feed exactly chunk_samples to maintain constant right-context (which is 0)
        # NeMo's buffer requires consistent right-context across all feeds
        if not self._primed:
            if processable.shape[1] < chunk_samples:
                # Not enough to prime
                if chunk.is_final:
                    # Not enough audio to safely prime; treat as empty final
                    logger.debug("[%s] final before prime (have=%d < need=%d): finalize empty", self.conn_id, int(processable.shape[1]), chunk_samples)
                    # Reset session state for next utterance
                    self._reset_for_new_utterance()
                    return StreamResult(
                        conn_id=self.conn_id,
                        chunk_id=chunk.chunk_id,
                        text="",
                        delta="",
                        is_final=True,
                    )
                # Not final: wait for more
                self._pending = audio  # keep everything
                logger.debug(
                    "[%s] need prime (%d) but only have %d -> waiting",
                    self.conn_id,
                    chunk_samples,
                    int(processable.shape[1]),
                )
                return None
            remaining_after = processable[:, chunk_samples:]
            to_feed = processable[:, :chunk_samples]
            is_last_chunk = chunk.is_final and remaining_after.shape[1] == 0 and tail.numel() == 0
            processable = remaining_after
            logger.debug(
                "[%s] PRIME feed: %d samples, remain=%d",
                self.conn_id,
                chunk_samples,
                int(processable.shape[1]),
            )

            text = _feed_buffer(to_feed, is_last_chunk)
            if text is not None:
                latest_text = text
            self._primed = True

        # Feed fixed-size chunk blocks
        while processable.shape[1] >= chunk_samples:
            remaining_after = processable[:, chunk_samples:]
            to_feed = processable[:, :chunk_samples]
            is_last_chunk = chunk.is_final and remaining_after.shape[1] == 0 and tail.numel() == 0
            processable = remaining_after
            logger.debug(
                "[%s] CHUNK feed: %d, remain=%d",
                self.conn_id,
                chunk_samples,
                int(processable.shape[1]),
            )

            text = _feed_buffer(to_feed, is_last_chunk)
            if text is not None:
                latest_text = text

        # If this is a final chunk, flush any remaining full-frame audio even if less than chunk
        if chunk.is_final and processable.shape[1] > 0:
            to_feed = processable
            processable = processable[:, 0:0]
            logger.debug(
                "[%s] FINAL feed: %d (remaining frames)",
                self.conn_id,
                int(to_feed.shape[1]),
            )
            text = _feed_buffer(to_feed, True)
            if text is not None:
                latest_text = text

        # Whatever remains after feeding full steps plus the non-frame-aligned tail becomes pending
        if chunk.is_final:
            self._pending = None
        else:
            if processable.shape[1] > 0:
                self._pending = torch.cat([processable, tail], dim=1) if tail.numel() > 0 else processable
            else:
                self._pending = tail if tail.numel() > 0 else None
        logger.debug(
            "[%s] end of consume: pending=%d, latest_text_len=%d",
            self.conn_id,
            0 if self._pending is None else int(self._pending.shape[1]),
            0 if latest_text is None else len(latest_text),
        )

        # On final, emit at least a final boundary even if no new text
        if latest_text is None and not chunk.is_final:
            return None

        # Suppress duplicate emissions when hypothesis hasn't changed
        if latest_text == self.last_text and not chunk.is_final:
            logger.debug("[%s] suppress duplicate hypothesis: '%s'", self.conn_id, latest_text)
            return None

        if latest_text is None:
            latest_text = self.last_text

        if latest_text.startswith(self.last_text):
            delta = latest_text[len(self.last_text) :]
        else:
            delta = latest_text
        logger.debug("[%s] new text: '%s' (delta='%s')", self.conn_id, latest_text, delta)
        self.last_text = latest_text

        is_final = bool(chunk.is_final)
        result = StreamResult(
            conn_id=self.conn_id,
            chunk_id=chunk.chunk_id,
            text=latest_text,
            delta=delta,
            is_final=is_final,
        )
        
        # If final, reset internal state for next utterance
        if is_final:
            self._reset_for_new_utterance()

        return result

    def _decode_and_update(self) -> Optional[str]:
        engine = self.engine
        buffer = self.buffer

        input_signal = buffer.samples
        if input_signal.numel() == 0:
            return None
        input_lengths = buffer.context_size_batch.total()
        logger.debug(
            "[%s] decode: buffer samples=%d, lengths=%s",
            self.conn_id,
            int(input_signal.shape[1]),
            str(input_lengths.tolist() if hasattr(input_lengths, 'tolist') else input_lengths),
        )
        encoder_output, _ = engine.model(
            input_signal=input_signal,
            input_signal_length=input_lengths,
        )
        encoder_output = encoder_output.transpose(1, 2)

        encoder_context = buffer.context_size.subsample(engine.encoder_frame2audio_samples)
        encoder_context_batch = buffer.context_size_batch.subsample(engine.encoder_frame2audio_samples)
        encoder_output = encoder_output[:, encoder_context.left :]
        logger.debug(
            "[%s] decode: enc_out=%s, ctx_frames L=%d C=%d R=%d",
            self.conn_id,
            tuple(encoder_output.shape),
            encoder_context.left,
            encoder_context.chunk,
            encoder_context.right,
        )

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
        text = engine.model.tokenizer.ids_to_text(tokens)
        logger.debug("[%s] decode: tokens=%d -> '%s'", self.conn_id, len(tokens), text)
        return text


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

        def _frames_nonzero(sec: float) -> int:
            frames = int(sec * features_per_sec / subsampling_factor)
            return max(frames, 1)

        def _frames_allow_zero(sec: float) -> int:
            frames = int(sec * features_per_sec / subsampling_factor)
            return max(frames, 0)

        self.context_encoder_frames = ContextSize(
            left=_frames_nonzero(STREAM_LEFT_CONTEXT_SECS),
            chunk=_frames_nonzero(STREAM_CHUNK_SECS),
            right=_frames_allow_zero(STREAM_RIGHT_CONTEXT_SECS),
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
        logger.info(
            "Encoder frame: %d samples (%.1f ms at %d Hz)",
            self.encoder_frame2audio_samples,
            1000.0 * self.encoder_frame2audio_samples / float(self.sample_rate),
            self.sample_rate,
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
