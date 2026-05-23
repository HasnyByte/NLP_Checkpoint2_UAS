import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import librosa
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TEMP_DIR = PROJECT_ROOT / "temp"

TARGET_SR = 16000
TRIM_TOP_DB = 30
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


@dataclass
class AudioProcessingResult:
    source_path: str
    output_path: str
    status: str
    sample_rate: int
    duration_s: float
    noise_removed: bool
    message: str = ""


def ensure_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    base = Path(name).stem
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "audio"


def scan_audio_files(root_dir: str | os.PathLike[str]) -> List[Path]:
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Folder audio '{root_path}' tidak ditemukan.")

    files = []
    for path in root_path.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    files.sort(key=lambda p: p.name.lower())
    return files


def _load_audio(audio_path: Path):
    y, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True, res_type="soxr_hq")
    return y, sr


def _trim_and_normalize(y: np.ndarray) -> tuple[np.ndarray, bool]:
    trimmed_y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    noise_removed = len(trimmed_y) != len(y)

    if trimmed_y.size == 0:
        trimmed_y = y

    peak = float(np.max(np.abs(trimmed_y)))
    if peak > 0:
        trimmed_y = trimmed_y / peak

    if not np.isfinite(trimmed_y).all():
        trimmed_y = np.nan_to_num(trimmed_y, nan=0.0, posinf=0.0, neginf=0.0)

    return trimmed_y, noise_removed


def preprocess_audio(
    input_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
) -> AudioProcessingResult:
    """Preprocess audio untuk STT: resample, trim silence, normalisasi, simpan WAV 16kHz mono."""
    ensure_directories()

    source_path = Path(input_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Audio tidak ditemukan: {source_path}")

    output_root = Path(output_dir) if output_dir is not None else PROCESSED_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    base_name = sanitize_filename(source_path.name)
    output_path = output_root / f"{base_name}.wav"

    try:
        y, sr = _load_audio(source_path)
        cleaned_y, noise_removed = _trim_and_normalize(y)
        duration_s = float(len(cleaned_y) / TARGET_SR)

        sf.write(str(output_path), cleaned_y, TARGET_SR, subtype="PCM_16")
        return AudioProcessingResult(
            source_path=str(source_path),
            output_path=str(output_path),
            status="success",
            sample_rate=TARGET_SR,
            duration_s=round(duration_s, 3),
            noise_removed=noise_removed,
        )
    except Exception as exc:
        return AudioProcessingResult(
            source_path=str(source_path),
            output_path=str(output_path),
            status="failed",
            sample_rate=TARGET_SR,
            duration_s=0.0,
            noise_removed=False,
            message=str(exc),
        )


def preload_audio_bytes(input_path: str | os.PathLike[str], output_dir: str | os.PathLike[str] | None = None) -> tuple[AudioProcessingResult, bytes]:
    result = preprocess_audio(input_path, output_dir)
    if result.status != "success":
        raise RuntimeError(result.message or f"Gagal preprocessing {input_path}")

    with open(result.output_path, "rb") as file:
        payload = file.read()
    return result, payload


def preprocess_batch(root_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str] | None = None) -> List[AudioProcessingResult]:
    results = []
    for audio_path in scan_audio_files(root_dir):
        results.append(preprocess_audio(audio_path, output_dir))
    return results


def format_processing_report(results: Iterable[AudioProcessingResult]) -> str:
    results = list(results)
    success = sum(1 for item in results if item.status == "success")
    failed = len(results) - success
    total_duration = round(sum(item.duration_s for item in results if item.status == "success"), 2)
    return (
        f"Total file: {len(results)} | Sukses: {success} | Gagal: {failed} | Durasi: {total_duration:.2f}s"
    )
