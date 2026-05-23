import os
import subprocess
import sys
import uuid

try:
    from .utils import TEMP_DIR, normalize_text
except ImportError:
    from utils import TEMP_DIR, normalize_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_env_coqui = os.getenv("COQUI_DIR")
if _env_coqui:
    COQUI_DIR = _env_coqui
else:
    # prefer existing coqui_utils folder (supplied in repo), then coqui_tts
    cand_utils = os.path.join(BASE_DIR, "coqui_utils")
    cand_tts = os.path.join(BASE_DIR, "coqui_tts")
    if os.path.exists(cand_utils):
        COQUI_DIR = cand_utils
    else:
        COQUI_DIR = cand_tts

COQUI_MODEL_PATH = os.getenv("COQUI_MODEL_PATH", os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth"))
COQUI_CONFIG_PATH = os.getenv("COQUI_CONFIG_PATH", os.path.join(COQUI_DIR, "config.json"))
COQUI_SPEAKER = os.getenv("COQUI_SPEAKER", "wibowo")
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", "180"))
DEFAULT_TTS_BINARY = os.path.join(os.path.dirname(sys.executable), "tts")
TTS_BINARY = os.getenv("TTS_BINARY", DEFAULT_TTS_BINARY)


def _validate_coqui_paths() -> None:
    if not os.path.exists(COQUI_MODEL_PATH):
        raise FileNotFoundError(f"Model Coqui TTS tidak ditemukan: {COQUI_MODEL_PATH}")
    if not os.path.exists(COQUI_CONFIG_PATH):
        raise FileNotFoundError(f"Config Coqui TTS tidak ditemukan: {COQUI_CONFIG_PATH}")


def transcribe_text_to_speech(text: str) -> str:
    text = normalize_text(text)
    if not text:
        raise ValueError("Teks kosong, tidak bisa dibuat menjadi audio.")
    return _tts_with_coqui(text)


def _tts_with_coqui(text: str) -> str:
    _validate_coqui_paths()
    output_path = os.path.join(TEMP_DIR, f"tts_{uuid.uuid4()}.wav")

    cmd = [
        TTS_BINARY,
        "--text", text,
        "--model_path", COQUI_MODEL_PATH,
        "--config_path", COQUI_CONFIG_PATH,
        "--out_path", output_path,
    ]

    if COQUI_SPEAKER:
        cmd.extend(["--speaker_idx", COQUI_SPEAKER])

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=TTS_TIMEOUT,
            cwd=COQUI_DIR,
        )
        if result.stderr:
            print(result.stderr)
    except subprocess.TimeoutExpired:
        raise RuntimeError("TTS timeout. Coba gunakan teks respons yang lebih pendek.")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"TTS subprocess failed: {exc.stderr or exc}")

    return output_path
