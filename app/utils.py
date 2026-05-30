import os
import re
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

LLM_FALLBACK_RESPONSE = "Maaf, saya belum bisa memberikan jawaban yang sesuai."
STT_EMPTY_MESSAGE = "Maaf, suara belum terdengar jelas. Silakan ulangi rekaman."

_ID_DIGITS = {
    "nol": 0,
    "kosong": 0,
    "satu": 1,
    "dua": 2,
    "tiga": 3,
    "empat": 4,
    "lima": 5,
    "enam": 6,
    "tujuh": 7,
    "delapan": 8,
    "sembilan": 9,
}
_ID_NUMBER_WORDS = set(_ID_DIGITS) | {
    "sepuluh",
    "sebelas",
    "belas",
    "puluh",
    "seratus",
    "ratus",
    "seribu",
    "ribu",
}
_ID_NUMBER_SCALES = {"sepuluh", "sebelas", "belas", "puluh", "seratus", "ratus", "seribu", "ribu"}

_TTS_ABBREVIATIONS = {
    "AI": "a i",
    "API": "a pi ai",
    "LLM": "el el em",
    "STT": "es te te",
    "TTS": "te te es",
    "NLP": "en el pi",
    "PDF": "pe de ef",
}
_TTS_PRONUNCIATION_MAP = {
    "saya": "saya",
    "mengalami": "meng alami",
    "kesulitan": "kesulitan",
    "memproses": "mem proses",
    "pertanyaan": "per tanyaan",
    "silakan": "silahkan",
    "lagi": "lagi",
    "respons": "respon",
    "pipeline": "paip lain",
    "audio": "audio",
}


def normalize_text(text: str) -> str:
    """Normalisasi sederhana sebelum teks dikirim ke LLM."""
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", text)
    return text


def _parse_under_100(words: list[str]) -> int:
    if not words:
        return 0
    if words[0] == "sepuluh":
        return 10 + _parse_under_100(words[1:])
    if words[0] == "sebelas":
        return 11 + _parse_under_100(words[1:])
    if "belas" in words:
        index = words.index("belas")
        return (_parse_under_100(words[:index]) or 1) + 10 + _parse_under_100(words[index + 1:])
    if "puluh" in words:
        index = words.index("puluh")
        return (_parse_under_100(words[:index]) or 1) * 10 + _parse_under_100(words[index + 1:])
    return sum(_ID_DIGITS.get(word, 0) for word in words)


def _parse_under_1000(words: list[str]) -> int:
    if not words:
        return 0
    if words[0] == "seratus":
        return 100 + _parse_under_100(words[1:])
    if "ratus" in words:
        index = words.index("ratus")
        return (_parse_under_100(words[:index]) or 1) * 100 + _parse_under_100(words[index + 1:])
    return _parse_under_100(words)


def _parse_indonesian_number(words: list[str]) -> int:
    if not words:
        return 0
    if words[0] == "seribu":
        return 1000 + _parse_under_1000(words[1:])
    if "ribu" in words:
        index = words.index("ribu")
        return (_parse_under_1000(words[:index]) or 1) * 1000 + _parse_under_1000(words[index + 1:])
    return _parse_under_1000(words)


def _convert_indonesian_number_words(text: str) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    converted: list[str] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        lowered = token.lower()

        if not re.fullmatch(r"\w+", token, flags=re.UNICODE) or lowered not in _ID_NUMBER_WORDS:
            converted.append(token)
            index += 1
            continue

        phrase: list[str] = []
        while index < len(tokens):
            lowered = tokens[index].lower()
            if re.fullmatch(r"\w+", tokens[index], flags=re.UNICODE) and lowered in _ID_NUMBER_WORDS:
                phrase.append(lowered)
                index += 1
            else:
                break

        if len(phrase) > 1 and not any(word in _ID_NUMBER_SCALES for word in phrase):
            converted.extend(str(_ID_DIGITS[word]) for word in phrase)
        else:
            converted.append(str(_parse_indonesian_number(phrase)))

    result = " ".join(converted)
    result = re.sub(r"\s+([,.!?;:])", r"\1", result)
    return result


def normalize_transcript_text(text: str) -> str:
    """Normalisasi hasil STT, termasuk kata bilangan Bahasa Indonesia."""
    text = normalize_text(text)
    if not text:
        return ""
    return normalize_text(_convert_indonesian_number_words(text))


def prepare_text_for_tts(text: str, fallback: str = LLM_FALLBACK_RESPONSE) -> str:
    """Bersihkan teks agar lebih stabil dibaca mesin TTS."""
    text = normalize_text(text)
    if not text:
        text = fallback

    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", " ", text)
    text = re.sub(r"[*_#>\[\]{}|~^]+", " ", text)
    text = re.sub(r"^\s*[-+•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()

    for source, replacement in _TTS_ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(source)}\b", replacement, text)

    for source, replacement in _TTS_PRONUNCIATION_MAP.items():
        text = re.sub(rf"\b{re.escape(source)}\b", replacement, text, flags=re.IGNORECASE)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned_sentences = []
    total_chars = 0
    for sentence in sentences:
        sentence = sentence.strip(" -")
        if not sentence:
            continue
        if len(sentence) > 170:
            chunks = re.split(r"(?<=,)\s+|\s+(?=dan|atau|karena|tetapi|namun)\b", sentence)
        else:
            chunks = [sentence]
        for chunk in chunks:
            chunk = normalize_text(chunk)
            if not chunk:
                continue
            if total_chars + len(chunk) > 420:
                break
            cleaned_sentences.append(chunk)
            total_chars += len(chunk)

    text = ". ".join(sentence.rstrip(".") for sentence in cleaned_sentences)
    if text and not re.search(r"[.!?]$", text):
        text += "."

    return normalize_text(text or fallback)


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
