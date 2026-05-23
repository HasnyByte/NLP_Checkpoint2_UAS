import os
import re
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


def normalize_text(text: str) -> str:
    """Normalisasi sederhana sebelum teks dikirim ke LLM."""
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", text)
    return text


def safe_delete(path: Optional[str]) -> None:
    """Hapus file sementara tanpa menghentikan aplikasi jika gagal."""
    if not path:
        return
    try:
        if os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
    except Exception as exc:
        print(f"[WARNING] Gagal menghapus file sementara {path}: {exc}")


def get_file_ext(filename: str, default: str = ".wav") -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext else default
