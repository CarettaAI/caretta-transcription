from __future__ import annotations
import asyncio
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status, Request, Form

from .audio import ensure_mono_16k, schedule_cleanup
from .model import _to_builtin
from .schemas import TranscriptionResponse
from .config import logger
from .audio import bytes_to_chunks
from parakeet_service.types import AudioChunk
from parakeet_service.model import transcribe_stream_chunks
import io, uuid
import soundfile as sf
import torch
import numpy as np
import torchaudio.functional as AF
from .config import TARGET_SR

from parakeet_service.model import reset_fast_path
from parakeet_service.chunker import vad_chunk_lowmem, vad_chunk_streaming


router = APIRouter(tags=["speech"])


@router.get("/healthz", summary="Liveness/readiness probe")
def health():
    return {"status": "ok"}


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file",
)
@router.post(
    "/audio/transcriptions",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file",
)
async def transcribe_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., media_type="audio/*"),
    include_timestamps: bool = Form(
        False, description="Return char/word/segment offsets",
    ),
    should_chunk: bool = Form(True,
        description="If true (default), split long audio into "
                    "~60s VAD-aligned chunks for batching"),
):
    # Create temp file name (used only by fallback). Prefer in-memory fast-path.
    suffix = Path(file.filename or "").suffix or ".wav"

    # Read whole upload into memory (we'll fall back to disk if anything fails)
    uploaded_bytes = await file.read()

    mp3_tmp_path = None
    tmp_path = None

    # Attempt in-memory processing first
    try:
        # Convert MP3 → WAV in-memory via ffmpeg if needed
        if suffix.lower() == ".mp3":
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-v", "error", "-nostdin", "-i", "pipe:0",
                "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                "-f", "wav", "pipe:1",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate(input=uploaded_bytes)
            if proc.returncode != 0:
                stderr_text = (stderr or b"").decode(errors="ignore")
                logger.error("FFmpeg failed: %s", stderr_text)
                raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                                    detail=f"Invalid audio format: {stderr_text[:200]}")
            wav_bytes = stdout
        else:
            wav_bytes = uploaded_bytes

        # Fast-path: convert bytes → AudioChunk[] in-memory
        if should_chunk:
            chunks = bytes_to_chunks(wav_bytes)
            if not chunks:
                # No speech detected — make a single chunk for full audio
                raise RuntimeError("VAD produced no chunks")
        else:
            # Create single AudioChunk for entire file
            import soundfile as _sf
            data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != TARGET_SR:
                tensor = torch.from_numpy(data).unsqueeze(0)
                tensor = AF.resample(tensor, sr, TARGET_SR)
                data = tensor.squeeze(0).numpy()
            pcm16 = np.clip(data * 32768, -32768, 32767).astype(np.int16).tobytes()
            chunks = [AudioChunk(chunk_id=uuid.uuid4().hex, pcm16=pcm16, sample_rate=TARGET_SR)]

        logger.info("transcribe(): (fast-path) sending %d chunks to ASR", len(chunks))

        model = request.app.state.asr_model
        outs = transcribe_stream_chunks(model, chunks, batch_size=len(chunks))

        # Build response
        if isinstance(outs, tuple):
            outs = outs[0]
        texts = []
        ts_agg = [] if include_timestamps else None
        merged = defaultdict(list)
        for h in outs:
            texts.append(getattr(h, "text", str(h)))
            if include_timestamps:
                for k, v in _to_builtin(getattr(h, "timestamp", {})).items(): # type: ignore
                    merged[k].extend(v)

        merged_text = " ".join(texts).strip()
        timestamps = dict(merged) if include_timestamps else None
        return TranscriptionResponse(text=merged_text, timestamps=timestamps)
    except asyncio.CancelledError:
        # Clean up temporary files if processing was cancelled
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise
    except BrokenPipeError:
        logger.error("FFmpeg process terminated unexpectedly")
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audio processing failed due to FFmpeg crash"
        )
    except Exception as exc:
        logger.debug("In-memory fast-path failed, falling back to disk: %s", exc)
        # Fallback: write uploaded bytes to temp file and continue with original disk path
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_bytes)
            tmp_path = Path(tmp.name)
    finally:
        await file.close()

    # Process audio to ensure mono 16kHz
    original, to_model = ensure_mono_16k(tmp_path)

    if should_chunk:
        # Use low-memory chunker for non-streaming requests
      chunk_paths = vad_chunk_lowmem(to_model) or [to_model]
    else:
        chunk_paths = [to_model]

    logger.info("transcribe(): sending %d chunks to ASR", len(chunk_paths))

    # Clean up all temporary files
    cleanup_files = [original, to_model] + chunk_paths
    if mp3_tmp_path:
        cleanup_files.append(mp3_tmp_path)
    schedule_cleanup(background_tasks, *cleanup_files)

    # 2 – run ASR
    model = request.app.state.asr_model

    try:
        outs = model.transcribe(
            [str(p) for p in chunk_paths],
            batch_size=2,
            timestamps=include_timestamps,
        )
        if (
          not include_timestamps                     # switch back to model fast-path if timestamps turned off
          and getattr(model.cfg.decoding, "compute_timestamps", False)
        ):
          reset_fast_path(model)                    
    except RuntimeError as exc:
        logger.exception("ASR failed")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc

    if isinstance(outs, tuple):
      outs = outs[0]
    texts = []
    ts_agg = [] if include_timestamps else None
    merged = defaultdict(list)

    for h in outs:
        texts.append(getattr(h, "text", str(h)))
        if include_timestamps:
            for k, v in _to_builtin(getattr(h, "timestamp", {})).items(): # type: ignore
                merged[k].extend(v)           # concat lists

    merged_text = " ".join(texts).strip()
    timestamps  = dict(merged) if include_timestamps else None

    return TranscriptionResponse(text=merged_text, timestamps=timestamps)

@router.get("/debug/cfg")
def show_cfg(request: Request):
    from omegaconf import OmegaConf
    model = request.app.state.asr_model         
    yaml_str = OmegaConf.to_yaml(model.cfg, resolve=True) 
    return yaml_str