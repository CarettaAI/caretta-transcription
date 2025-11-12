import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List

from .config import STREAM_BATCH_WINDOW_MS, STREAM_MAX_BATCH, STREAM_QUEUE_CAPACITY
from .streaming_engine import StreamResult, StreamTask, StreamingEngine

logger = logging.getLogger("batcher")
logger.setLevel(logging.INFO)


transcription_queue: asyncio.Queue[StreamTask] = asyncio.Queue(maxsize=STREAM_QUEUE_CAPACITY)
condition = asyncio.Condition()  # wakes websocket consumers
results: Dict[str, Deque[StreamResult]] = defaultdict(deque)


async def batch_worker(
    engine: StreamingEngine,
    batch_ms: float = STREAM_BATCH_WINDOW_MS,
    max_batch: int = STREAM_MAX_BATCH,
) -> None:
    """Drain queued streaming chunks, run inference off-thread, publish results."""

    logger.info(
        "worker started (batch ≤%d, window %.0f ms, sessions=%d)",
        max_batch,
        batch_ms,
        engine.active_session_count(),
    )

    while True:
        task = await transcription_queue.get()
        batch: List[StreamTask] = [task]

        deadline = time.monotonic() + batch_ms / 1000.0
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

        try:
            decoded = await asyncio.to_thread(engine.process_batch, batch)
        except Exception:  # pragma: no cover - defensive log
            logger.exception("Streaming batch execution failed")
            decoded = []
        finally:
            for _ in batch:
                transcription_queue.task_done()

        if not decoded:
            continue

        for item in decoded:
            results[item.conn_id].append(item)

        async with condition:
            condition.notify_all()
