"""
Global constants.
"""

from enum import IntEnum
import os

CLOUDFLARE_VERIFICATION_FAILED_MESSAGE = "PÄRINGU KONTROLL EBAÕNNESTUS. PALUN PROOVI UUESTI VÕI VÄRSKENDA LEHEKÜLGE!"

REPO_PATH = os.path.dirname(os.path.dirname(__file__))

# Survey Link URL (to be removed) #00729c
SURVEY_LINK = """<div style='text-align: left; margin: 20px 0;'>
    <div style='display: inline-block; border: 2px solid #00729c; padding: 20px; padding-bottom: 10px; padding-top: 10px; border-radius: 5px;'>
        <span style='color: #00729c; font-weight: bold;'>New Launch! Copilot Arena: <a href='https://marketplace.visualstudio.com/items?itemName=copilot-arena.copilot-arena' style='color: #00729c; text-decoration: underline;'>VS Code Extension</a> to compare Top LLMs</span>
    </div>
</div>"""
# SURVEY_LINK = ""

##### For the gradio web server
SERVER_ERROR_MSG = (
    "***Partneri tehisaru ei vastanud ootuspäraselt. Palun värskenda lehekülge ja proovime uuesti!***"
)
TEXT_MODERATION_MSG = (
    "$MODERATION$ YOUR TEXT VIOLATES OUR CONTENT MODERATION GUIDELINES."
)
IMAGE_MODERATION_MSG = (
    "$MODERATION$ YOUR IMAGE VIOLATES OUR CONTENT MODERATION GUIDELINES."
)
MODERATION_MSG = "SISEND EI VASTA MODEREERIMISREEGLITELE."
CONVERSATION_LIMIT_MSG = "VESTLUSE PIKKUSE LIMIIT ON TÄIS. PALUN ALUSTA UUT VESTLUST."
INACTIVE_MSG = "SESSIOON ON AEGUNUD. PALUN VÄRSKENDA LEHEKÜLGE."
SLOW_MODEL_MSG = (
    "⚠️  Mudelid mõtlevad. Palun ole kannatlik — see võtab natukene aega."
)
RATE_LIMIT_MSG = "**RATE LIMIT OF THIS MODEL IS REACHED. PLEASE COME BACK LATER OR USE <span style='color: red; font-weight: bold;'>[BATTLE MODE](https://lmarena.ai)</span> (the 1st tab).**"
# Maximum input length
INPUT_CHAR_LEN_LIMIT = int(os.getenv("FASTCHAT_INPUT_CHAR_LEN_LIMIT", 12000))
BLIND_MODE_INPUT_CHAR_LEN_LIMIT = int(
    os.getenv("FASTCHAT_BLIND_MODE_INPUT_CHAR_LEN_LIMIT", 30000)
)
# Maximum conversation turns
CONVERSATION_TURN_LIMIT = 50
# Session expiration time
SESSION_EXPIRATION_TIME = 3600
# The output dir of log files
LOGDIR = os.getenv("LOGDIR", ".")
# CPU Instruction Set Architecture
CPU_ISA = os.getenv("CPU_ISA")


##### For the controller and workers (could be overwritten through ENV variables.)
CONTROLLER_HEART_BEAT_EXPIRATION = int(
    os.getenv("FASTCHAT_CONTROLLER_HEART_BEAT_EXPIRATION", 90)
)
WORKER_HEART_BEAT_INTERVAL = int(os.getenv("FASTCHAT_WORKER_HEART_BEAT_INTERVAL", 45))
WORKER_API_TIMEOUT = int(os.getenv("FASTCHAT_WORKER_API_TIMEOUT", 100))
WORKER_API_EMBEDDING_BATCH_SIZE = int(
    os.getenv("FASTCHAT_WORKER_API_EMBEDDING_BATCH_SIZE", 4)
)


class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006
