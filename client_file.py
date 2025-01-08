import asyncio
import json
import wave
import argparse
from urllib.parse import urlencode

import numpy as np
import websockets


class AudioStreamClient:
    def __init__(self, wav_path, chunk_duration_ms=10):
        self.wav_path = wav_path
        self.chunk_duration = chunk_duration_ms / 1000.0

    async def stream(self, uri):
        with wave.open(self.wav_path, "rb") as wav_file:
            sampling_rate = wav_file.getframerate()
            chunk_size = int(sampling_rate * self.chunk_duration)
            params = {"sampling_rate": str(sampling_rate), "encoding": "linear16"}
            full_uri = f"{uri}?{urlencode(params)}"

            async with websockets.connect(full_uri) as ws:
                try:
                    while True:
                        data = wav_file.readframes(chunk_size)
                        if not data:
                            await ws.send("EOF")
                            break
                        audio_array = np.frombuffer(data, dtype=np.int16)
                        if wav_file.getnchannels() > 1:
                            audio_array = (
                                audio_array.reshape(-1, wav_file.getnchannels())
                                .mean(axis=1)
                                .astype(np.int16)
                            )

                        chunk = audio_array.tobytes()
                        await ws.send(chunk)

                        try:
                            response = await asyncio.wait_for(ws.recv(), timeout=0.01)
                            result = json.loads(response)
                            text = result.get("text")
                            print(text, end=" ", flush=True)
                        except asyncio.TimeoutError:
                            continue
                    try:
                        final_response = await asyncio.wait_for(ws.recv(), timeout=10)
                        print(final_response["text"])
                    except asyncio.TimeoutError:
                        print("Done streaming and connection cloes")
                except websockets.ConnectionClosed:
                    print("\n\n***Connection closed by server***")


async def main(uri, wav_path, chunk_duration_ms):
    client = AudioStreamClient(wav_path, chunk_duration_ms)

    await client.stream(uri)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket Client")
    parser.add_argument("--uri", default="ws://0.0.0.0:8000", help="URI at which server is running")
    parser.add_argument("--wav-path",default="examples/example1.wav", help="Wav file to read and send to server")
    parser.add_argument(
        "--chunk-duration", default=10, help="Lenght of chunks to send to server in ms"
    )
    args = parser.parse_args()

    asyncio.run(main(args.uri, args.wav_path, int(args.chunk_duration)))
