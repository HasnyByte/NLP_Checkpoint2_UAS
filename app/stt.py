import os
import subprocess
import sys
import uuid
from pathlib import Path

try:
    from .utils import TEMP_DIR, normalize_text, safe_delete
except ImportError:
    from utils import TEMP_DIR, normalize_text, safe_delete

try:
    from processing import preprocess_audio
except ImportError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from processing import preprocess_audio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

_env_whisper_dir = os.getenv("WHISPER_DIR")
if _env_whisper_dir:
    WHISPER_DIR = _env_whisper_dir
else:
    # prefer app/whisper.cpp when present in this repo layout, fallback to models/whisper.cpp
    cand_app = os.path.join(PROJECT_ROOT, "app", "whisper.cpp")
    cand_models = os.path.join(PROJECT_ROOT, "models", "whisper.cpp")
    if os.path.exists(cand_app):
        WHISPER_DIR = cand_app
    else:
        WHISPER_DIR = cand_models
WHISPER_BINARY = os.getenv("WHISPER_BINARY", os.path.join(WHISPER_DIR, "build", "bin", "whisper-cli"))
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH")
if not WHISPER_MODEL_PATH:
    # Prefer common model filenames if present (base first), fallback to large-v3-turbo
    candidates = [
        os.path.join(WHISPER_DIR, "models", "ggml-base.bin"),
        os.path.join(WHISPER_DIR, "models", "ggml-base.en.bin"),
        os.path.join(WHISPER_DIR, "models", "ggml-large-v3-turbo.bin"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            WHISPER_MODEL_PATH = cand
            break
    else:
        WHISPER_MODEL_PATH = os.path.join(WHISPER_DIR, "models", "ggml-large-v3-turbo.bin")
WHISPER_TIMEOUT = int(os.getenv("WHISPER_TIMEOUT", "180"))

if os.name == "nt" and not WHISPER_BINARY.endswith(".exe"):
    WHISPER_BINARY += ".exe"


def _validate_whisper_paths() -> None:
    if not os.path.exists(WHISPER_BINARY):
        raise FileNotFoundError(f"Binary whisper-cli tidak ditemukan: {WHISPER_BINARY}")
    if not os.path.exists(WHISPER_MODEL_PATH):
        raise FileNotFoundError(f"Model Whisper tidak ditemukan: {WHISPER_MODEL_PATH}")


def transcribe_audio_file(audio_path: str) -> str:
    """Transkrip audio dari path file menggunakan whisper.cpp CLI."""
    _validate_whisper_paths()

    output_base = os.path.join(TEMP_DIR, f"transcription_{uuid.uuid4()}")
    output_txt = f"{output_base}.txt"
    preprocessed_path = None

    try:
        processed = preprocess_audio(audio_path, output_dir=TEMP_DIR)
        if processed.status == "success":
            preprocessed_path = processed.output_path
            audio_input = preprocessed_path
        else:
            audio_input = audio_path

        cmd = [
            WHISPER_BINARY,
            "-m", WHISPER_MODEL_PATH,
            "-l", "auto",
            "-f", audio_input,
            "-otxt",
            "-of", output_base,
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=WHISPER_TIMEOUT,
            )
            if result.stderr:
                print(result.stderr)
        except subprocess.TimeoutExpired:
            return "[ERROR] Whisper timeout. Audio terlalu panjang atau model terlalu berat."
        except subprocess.CalledProcessError as exc:
            return f"[ERROR] Whisper failed: {exc.stderr or exc}"

        try:
            with open(output_txt, "r", encoding="utf-8") as file:
                return normalize_text(file.read())
        except FileNotFoundError:
            return "[ERROR] File hasil transkripsi tidak ditemukan."
    finally:
        safe_delete(output_txt)
        safe_delete(preprocessed_path)


def transcribe_speech_to_text(file_bytes: bytes, file_ext: str = ".wav") -> str:
    """Simpan bytes audio sementara, transkrip, lalu bersihkan file upload."""
    audio_path = os.path.join(TEMP_DIR, f"upload_{uuid.uuid4()}{file_ext}")
    try:
        with open(audio_path, "wb") as file:
            file.write(file_bytes)
        return transcribe_audio_file(audio_path)
    finally:
        safe_delete(audio_path)
