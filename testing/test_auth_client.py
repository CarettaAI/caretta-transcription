"""
Example WebSocket client with JWT authentication.

This demonstrates how to connect to the /ws endpoint with JWT token authentication.
The token can be passed either as a query parameter or in the Authorization header.
"""
import asyncio
import websockets
import json

# Replace with your actual JWT token from Supabase
JWT_TOKEN = "your-jwt-token-here"

# WebSocket URL - choose one of these methods:
# Method 1: Token in query parameter
WS_URL = f"ws://localhost:8000/ws?token={JWT_TOKEN}"

# Method 2: Token in Authorization header (shown below)
# WS_URL = "ws://localhost:8000/ws"

async def test_websocket():
    """Connect to WebSocket with authentication and send test audio."""
    
    # For header-based auth, pass extra_headers
    # headers = {"Authorization": f"Bearer {JWT_TOKEN}"}
    # async with websockets.connect(WS_URL, extra_headers=headers) as websocket:
    
    # For query parameter auth (simpler):
    try:
        async with websockets.connect(WS_URL) as websocket:
            print("✓ Connected successfully")
            
            # Send some test PCM audio data (16-bit int16 mono @ 16kHz)
            # This is just a placeholder - replace with actual audio data
            test_audio = b'\x00\x00' * 16000  # 1 second of silence
            
            await websocket.send(test_audio)
            print("✓ Sent audio data")
            
            # Receive transcription results
            async for message in websocket:
                data = json.loads(message)
                print(f"← {data}")
                
                if data.get("is_final"):
                    print(f"Final transcription: {data.get('text')}")
                    break
                    
    except websockets.exceptions.InvalidStatus as e:
        print(f"✗ Authentication failed: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
