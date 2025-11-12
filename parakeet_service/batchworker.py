import asyncio, logging, time, torch
from typing import List

from parakeet_service import model as mdl
from parakeet_service.types import AudioChunk

logger = logging.getLogger("batcher")
logger.setLevel(logging.DEBUG)

# -------- shared state -------------------------------------------------------
transcription_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
condition = asyncio.Condition()          # wakes websocket consumers
results: dict[str, str] = {}             # chunk_id -> text


# -------- main worker --------------------------------------------------------
async def batch_worker(model, batch_ms: float = 15.0, max_batch: int = 4):
    """Forever drain `transcription_queue` → ASR → `results`."""
    logger.info("worker started (batch ≤%d, window %.0f ms)", max_batch, batch_ms)
    logger.info("worker started with model id=%s", id(model))

    while True:
        first = await transcription_queue.get()      # blocks until 1st item
        batch: List[AudioChunk] = [first]

        # ---------- micro-batch gathering with timeout ----------
        deadline = time.monotonic() + batch_ms / 1000
        while len(batch) < max_batch:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                nxt = await asyncio.wait_for(transcription_queue.get(), timeout)
            except asyncio.TimeoutError:
                break
            else:
                batch.append(nxt)

        logger.debug("processing %d in-memory chunks", len(batch))

        # ---------- inference ----------
        try:
            with torch.inference_mode():
                outs = mdl.transcribe_stream_chunks(
                    model, batch, batch_size=len(batch)
                )
        except Exception as exc:
            logger.exception("ASR failed: %s", exc)
            for _ in batch:
                transcription_queue.task_done()
            continue

        # ---------- store results & notify ----------
        hyp_iter = iter(outs)
        for chunk in batch:
            try:
                hyp = next(hyp_iter)
            except StopIteration:
                logger.warning(
                    "ASR returned fewer hypotheses (%d) than chunks (%d)",
                    len(outs),
                    len(batch),
                )
                transcription_queue.task_done()
                continue

            results[chunk.chunk_id] = getattr(hyp, "text", str(hyp))
            transcription_queue.task_done()            # mark done

        async with condition:
            condition.notify_all()

        try:
            extra = next(hyp_iter)
        except StopIteration:
            extra = None
        if extra is not None:
            logger.warning(
                "ASR returned more hypotheses than requested; dropping extras"
            )
