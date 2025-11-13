import logging, os, sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"  # Keep hardcoded as requested

# Configuration from environment variables
TARGET_SR = int(os.getenv("TARGET_SR", "16000"))          # model’s native sample-rate
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "fp16")
DEVICE = os.getenv("DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "30"))   # seconds
PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", "60"))    # seconds

# Streaming configuration
STREAM_CHUNK_MS = int(os.getenv("STREAM_CHUNK_MS", "640"))              # target chunk duration (ms)
STREAM_LEFT_CONTEXT_MS = int(os.getenv("STREAM_LEFT_CONTEXT_MS", "9600"))
STREAM_RIGHT_CONTEXT_MS = int(os.getenv("STREAM_RIGHT_CONTEXT_MS", "0"))  # Set to 0 for consistent feeding
STREAM_MAX_UNFLUSHED_MS = int(os.getenv("STREAM_MAX_UNFLUSHED_MS", "1600"))
STREAM_BATCH_WINDOW_MS = float(os.getenv("STREAM_BATCH_WINDOW_MS", "20"))
STREAM_MAX_BATCH = int(os.getenv("STREAM_MAX_BATCH", "4"))
STREAM_QUEUE_CAPACITY = int(os.getenv("STREAM_QUEUE_CAPACITY", "512"))
STREAM_ENABLE_PARTIAL_FLUSH = os.getenv("STREAM_ENABLE_PARTIAL_FLUSH", "0").lower() in {"1", "true", "yes"}

STREAM_CHUNK_SECS = STREAM_CHUNK_MS / 1000.0
STREAM_LEFT_CONTEXT_SECS = STREAM_LEFT_CONTEXT_MS / 1000.0
STREAM_RIGHT_CONTEXT_SECS = STREAM_RIGHT_CONTEXT_MS / 1000.0
STREAM_MAX_UNFLUSHED_SECS = STREAM_MAX_UNFLUSHED_MS / 1000.0

# VAD / streaming audio params (configurable via .env)
# SAMPLE_RATE kept for backward-compatibility; defaults to TARGET_SR
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", str(TARGET_SR)))
# Default VAD window 640 samples (40 ms @ 16 kHz) to align with typical RNNT subsampling (4x @ 10ms)
# If you override this via env, ensure it's a multiple of the encoder frame size reported at service start.
VAD_WINDOW_SAMPLES = int(os.getenv("VAD_WINDOW_SAMPLES", "640"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.60"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "250"))
VAD_SPEECH_PAD_MS = int(os.getenv("VAD_SPEECH_PAD_MS", "120"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger("parakeet_service")
