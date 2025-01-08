import asyncio
import json
import argparse
from urllib.parse import urlencode
import websockets
import pyaudio

class AudioStreamClient:
    def __init__(self, device_index:int = 0, chunk_duration_ms:int =10):
        self.format = pyaudio.paInt16
        self.channels = 1
        self.sampling_rate = 16_000
        self.device_index = device_index
        self.chunk_duration = int(self.sampling_rate * (chunk_duration_ms / 1000))

    async def stream(self,uri):
        audio = pyaudio.PyAudio()
        stream = audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sampling_rate,
                input_device_index=self.device_index,
                input=True,
                frames_per_buffer=self.chunk_duration)
        
        print("*Recording*\n")
        
        params = {"sampling_rate": str(self.sampling_rate), "encoding": "linear16"}
        full_uri = f"{uri}?{urlencode(params)}"

        async with websockets.connect(full_uri) as ws:
            try:
                while True:
                    data = stream.read(self.chunk_duration,exception_on_overflow=False)

                    if not data:
                        await ws.send("EOF")
                        break

                    await ws.send(data)

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
                print("\n\n**Connection closed by server**\n\n")

        
async def main(uri,device_index,chunk_duration_ms):
    client = AudioStreamClient(device_index,chunk_duration_ms)

    await client.stream(uri)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket Client")
    parser.add_argument("--uri", default="ws://0.0.0.0:8000", help="URI at which server is running")
    parser.add_argument("--device-index",default=0, help="Microphone device index")
    parser.add_argument(
        "--chunk-duration", default=10, help="Lenght of chunks to send to server in ms"
    )
    args = parser.parse_args()

    asyncio.run(main(args.uri, int(args.device_index), int(args.chunk_duration)))

