import asyncio
import contextlib
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .batchworker import condition, results, transcription_queue
from .streaming_engine import StreamTask, StreamingEngine
from .streaming_vad import StreamingVAD

router = APIRouter()


@router.websocket("/ws")
async def ws_asr(ws: WebSocket):
    await ws.accept()

    engine: StreamingEngine = ws.app.state.streaming_engine  # type: ignore[attr-defined]
    conn_id = uuid.uuid4().hex
    engine.create_session(conn_id)
    results.pop(conn_id, None)  # ensure clean slate
    vad = StreamingVAD()

    async def producer() -> None:
        """Push VAD-produced chunks into the shared transcription queue."""
        try:
            while True:
                frame = await ws.receive_bytes()
                for chunk in vad.feed(frame):
                    await transcription_queue.put(StreamTask(conn_id=conn_id, chunk=chunk))
                    await ws.send_json({"status": "queued", "chunk_id": chunk.chunk_id})
        except WebSocketDisconnect:
            pass
        finally:
            vad.reset()

    async def consumer() -> None:
        """Stream decoder updates back to the client"""
        try:
            while True:
                async with condition:
                    await condition.wait()

                queue = results.get(conn_id)
                if not queue:
                    continue

                while queue:
                    item = queue.popleft()
                    await ws.send_json(
                        {
                            "chunk_id": item.chunk_id,
                            "text": item.text,
                            "delta": item.delta,
                            "is_final": item.is_final,
                        }
                    )
        except WebSocketDisconnect:
            pass

    try:
        await asyncio.gather(producer(), consumer())
    finally:
        engine.close_session(conn_id)
        results.pop(conn_id, None)
        with contextlib.suppress(Exception):
            await ws.close()
