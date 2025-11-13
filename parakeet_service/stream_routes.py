import asyncio
import contextlib
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .auth import verify_websocket_auth
from .batchworker import condition, results, transcription_queue
from .streaming_engine import StreamTask, StreamingEngine
from .streaming_vad import StreamingVAD
from .config import logger

router = APIRouter()


@router.websocket("/ws")
async def ws_asr(ws: WebSocket):
    # Verify authentication before accepting connection
    await verify_websocket_auth(ws)
    
    await ws.accept()

    engine: StreamingEngine = ws.app.state.streaming_engine  # type: ignore[attr-defined]
    conn_id = uuid.uuid4().hex
    engine.create_session(conn_id)
    logger.debug("[ws %s] connection opened", conn_id)
    results.pop(conn_id, None)  # ensure clean slate
    vad = StreamingVAD()

    async def producer() -> None:
        """Push VAD-produced chunks into the shared transcription queue."""
        try:
            while True:
                frame = await ws.receive_bytes()
                #logger.debug("[ws %s] recv frame: %d bytes", conn_id, len(frame))
                for chunk in vad.feed(frame):
                    await transcription_queue.put(StreamTask(conn_id=conn_id, chunk=chunk))
                    logger.debug("[ws %s] queued chunk: %s (%d samp) final=%s", conn_id, chunk.chunk_id, len(chunk), chunk.is_final)
                    await ws.send_json({"status": "queued", "chunk_id": chunk.chunk_id, "is_final": chunk.is_final})
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
                    logger.debug("[ws %s] sending result: chunk=%s, final=%s, text='%s'", conn_id, item.chunk_id, item.is_final, item.text)
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
        logger.debug("[ws %s] connection closed", conn_id)
        with contextlib.suppress(Exception):
            await ws.close()
