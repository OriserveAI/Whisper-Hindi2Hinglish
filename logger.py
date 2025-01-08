import logging
import os

loglevel = os.getenv("LOGLEVEL", "INFO")

logger = logging.getLogger("Whisper-Hindi2Hinglish")
logger.setLevel(loglevel)

console_formatter = logging.Formatter("%(name)s: %(levelname)s -> [%(filename)s:%(lineno)s - %(funcName)s] %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

