import argparse
import asyncio
import json
from functools import partial
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import webrtcvad
import websockets
from websockets.server import WebSocketServerProtocol

from logger import logger
from utils import audio_pre_processor, load_pipe, torch_dtype_from_str

vad = webrtcvad.Vad(3)


class Server:
    def __init__(self):
        self.audio_queue = asyncio.Queue()
        self.text_queue = asyncio.Queue()

        self.silence_counter = 0
        self.max_silence_chunks = 10
        self.audio_found = False
        self.audio_counter = 0
        self.old_text = ""

    async def handle_connection(self, ws: WebSocketServerProtocol, conn_url: str):
        """
        @function handle_connection
        @description Handles WebSocket connections and processes incoming audio data.
        @param ws: WebSocket connection object
        @param conn_url: Connection URL from client
        """
        logger.info("Got connection from path %s", conn_url)

        parsed_conn_url = urlparse(conn_url)

        query_params = parse_qs(parsed_conn_url.query)

        sampling_rate = query_params.get("samplingRate", ["16_000"])
        self.sampling_rate = int(sampling_rate[0])
        self.min_audio_duration = self.sampling_rate * 2

        self.encoding = query_params.get("encoding", ["linear16"])[0]

        async def receive_client_data():
            """
            @function receiver
            @description Receives audio data from the WebSocket and adds it to the audio queue.
            """
            async for message in ws:
                self.audio_queue.put_nowait(message)

        async def process_audio_to_text():
            """
            @function text_fetch
            @description Processes audio data, detects speech, and runs the model to generate text.
            """
            full_audio = np.array([])

            while True:
                data = await self.audio_queue.get()

                if isinstance(data, str) and data == "EOF":
                    if len(full_audio) > 0:
                        output = self.model(full_audio)
                        text = output["text"].strip()
                        logger.info("Recognised Output: %s", text)
                        if text and text != self.old_text:
                            text = text.replace("nan","")
                            self.old_text = text
                            ws_out = {"text": text}
                            await ws.send(json.dumps(ws_out))
                    await ws.close()

                audio, speech_present = audio_pre_processor(
                    data, self.sampling_rate, self.encoding, vad
                )
                full_audio = np.concatenate([full_audio, audio])

                if speech_present:
                    self.silence_counter = 0
                    self.audio_found = True
                else:
                    self.silence_counter += 1

                if len(full_audio) >= self.min_audio_duration or (
                    self.silence_counter >= self.max_silence_chunks and self.audio_found
                ):
                    self.audio_found = False
                    self.silence_counter = 0
                    output = self.model(full_audio)
                    text = output["text"].strip()
                    logger.info("Recognised Output: %s", text)
                    self.text_queue.put_nowait(text)
                    full_audio = np.array([], dtype=np.float32)

        async def send_text_response():
            """
            @function sender
            @description Sends recognized text from the text queue to the WebSocket.
            """
            while True:
                text = await self.text_queue.get()
                if text and text != self.old_text:
                    text = text.replace("nan","")
                    self.old_text = text
                    ws_out = {"text": text}

                    await ws.send(json.dumps(ws_out))

        await asyncio.gather(
            receive_client_data(), process_audio_to_text(), send_text_response()
        )

    async def init_server(
        self, host: str, port: str, model_id: str, device: str, dtype: torch.dtype
    ):
        """
        @function run_server
        @description Starts the WebSocket server and loads the model.
        @param host: Server host address
        @param port: Server port
        @param model_id: Model identifier
        @param device: Device to run the model on
        @param dtype: Data type for model computation
        """
        logger.info("Loading model %s", model_id)
        self.model = load_pipe(model_id, device, dtype)
        logger.info(f"Starting WebSocket server on ws://{host}:{port}")

        async with websockets.server.serve(
            partial(self.handle_connection), host, port
        ) as server:
            logger.info("Server is ready to accept connections...")
            await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument(
        "--model-id", default="Oriserve/Whisper-Hindi2Hinglish-Swift", help="Model ID"
    )
    parser.add_argument("--device", default="cuda", help="Device to run the model on")
    parser.add_argument(
        "--dtype", default="float16", help="Data type to run the model on"
    )

    args = parser.parse_args()

    dtype = torch_dtype_from_str(args.dtype, args.device)

    server = Server()
    asyncio.run(
        server.init_server(args.host, args.port, args.model_id, args.device, dtype)
    )
