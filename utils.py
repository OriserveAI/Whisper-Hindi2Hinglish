import audioop
from typing import Tuple

import librosa
import numpy as np
import torch
import webrtcvad
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def torch_dtype_from_str(dtype: str, device: str) -> torch.dtype:
    """
    @function torch_dtype_from_str
    @description Takes dtype object in string format and return a torch dtype object
    @param dtype: datatype in string format
    @param device: Device on which model is to run
    """
    if dtype == "float16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    else:
        if device == "cuda":
            return torch.float16
        else:
            return torch.float32


def load_pipe(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype,
) -> pipeline:
    """
    @function load_pipe
    @description Loads model using provided model_id and returns a huggingface pipeline object
    @param model_id: Model to load from huggingface
    @param device: Device to run model on
    @param dtype: Data type for model computation
    """
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"task": "transcribe", "language": "en"},
    )
    return pipe


def audio_pre_processor(
    audio: bytes, sr: int, encoding: str, vad: webrtcvad.Vad, target_sr: int = 16000
) -> Tuple[np.ndarray, bool]:
    """
    @function audio_pre_processor
    @description Preprocess audio data received from client convert it to a numpy array as well as check for speech presence
    @param audio: audio bytes received from client
    @param sr: sampling rate of the audio received
    @param encoding: encoding of the audio sent
    @param vad: webrtcvad vad object to check for speech presence
    """
    if encoding == "mulaw":
        audio = audioop.ulaw2in(audio, 2)

    try:
        is_speech_present = vad.is_speech(audio, sr)
    except Exception:
        is_speech_present = False

    array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16_000:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
    return array, is_speech_present
