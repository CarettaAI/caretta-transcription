from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from parakeet_service.streaming_vad import StreamingVAD
from parakeet_service.batchworker        import transcription_queue, condition, results
import asyncio
router = APIRouter()

@router.websocket("/ws")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    vad = StreamingVAD()

    async def producer():
        """push chunks into the global transcription queue"""
        try:
            while True:
                frame = await ws.receive_bytes()
                for chunk in vad.feed(frame):
                    await transcription_queue.put(chunk)
                    await ws.send_json({"status": "queued", "chunk_id": chunk.chunk_id})
        except WebSocketDisconnect:
            pass

    async def consumer():
        """stream results back as soon as they’re ready"""
        while True:
            async with condition:
                await condition.wait()          
            flushed = []
            for p, txt in list(results.items()):
                await ws.send_json({"chunk_id": p, "text": txt})
                flushed.append(p)
            for p in flushed:
                results.pop(p, None)

    await asyncio.gather(producer(), consumer())
