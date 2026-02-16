from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Plain transcription.")
    timestamps: Optional[Dict[str, Any]] = Field(
        None,
        description="Word/segment/char offsets (see NeMo docs).",
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="Time taken to process the audio in milliseconds.",
    )
