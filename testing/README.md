This folder contains a tiny test frontend to exercise the WebSocket streaming endpoint.

Files
- `index.html` — Opens a WebSocket to `/ws`, captures the microphone, downsamples to 16 kHz and sends 16-bit PCM frames as binary messages. It also displays JSON text messages returned by the server (partial transcripts).

How to use (quick):
1. Start the FastAPI server locally (see project README or below).
2. Open `testing/index.html` in a modern browser (Chrome/Edge/Firefox). If you get mixed-content errors, use `http://`/`ws://` to match the server.

Notes
- The client downsamples in JS with a simple averaging filter. It's not as high-quality as a proper resampler but is adequate for testing.
- Ensure the server expects 16 kHz int16 PCM (default config). If you changed SAMPLE_RATE or SAMPLE_RATE in `.env`, edit the `SAMPLE_RATE` constant inside `index.html` to match.

Running the server locally (powershell example)
---------------------------------------------
# create and activate venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies (preferably inside venv)
pip install -r requirements.txt
# If you plan to run on CPU-only, export DEVICE=cpu in your .env or set in the environment

# run the app
uvicorn parakeet_service.main:app --host 0.0.0.0 --port 8000 --log-level info

Important tips
- The model loads on startup and may take some time and lots of RAM/GPU. For local testing set `DEVICE=cpu` in your `.env` if you don't have a GPU.
- If running into CORS or mixed-content errors, make sure you serve the HTML via `http://` and the WS uses `ws://` (or both `https` and `wss`).
- For production use, use an orchestrator (k8s) and set `DEVICE=cuda` for GPU nodes.

If you'd like, I can:
- Add a tiny HTTP static server to serve `testing/index.html` so you can open `http://localhost:8000/test` and avoid file:// issues.
- Improve client resampling quality using an AudioWorklet.
- Provide an automated smoke-test (play an audio file through the WS and validate transcripts).
