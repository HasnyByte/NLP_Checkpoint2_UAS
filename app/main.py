import base64
import os
import re
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

try:
    from .llm import generate_response
    from .stt import transcribe_audio_file
    from .tts import transcribe_text_to_speech
    from .utils import TEMP_DIR, get_file_ext, normalize_text, safe_delete
except ImportError:
    from llm import generate_response
    from stt import transcribe_audio_file
    from tts import transcribe_text_to_speech
    from utils import TEMP_DIR, get_file_ext, normalize_text, safe_delete

app = FastAPI(title="Code-Switching Speech-to-Speech API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _analyze_language(text: str):
    """Heuristik yang lebih aman untuk menampilkan distribusi bahasa pada UI."""
    if not text:
        return {}, {}

    tokens = re.findall(r"[A-Za-z]+|[ء-ي]+", text.lower())
    if not tokens:
        return {}, {}

    ind_words = {
        "aku", "saya", "kamu", "anda", "kita", "mereka", "mau", "ingin", "boleh",
        "bisa", "tolong", "bantu", "mohon", "jadwal", "pesawat", "tiket", "penerbangan",
        "tanggal", "jam", "hari", "bulan", "tahun", "kapan", "siapa", "bagaimana", "dimana",
        "kemana", "darimana", "dari", "ke", "di", "dan", "atau", "yang", "untuk", "dengan",
        "sudah", "belum", "tidak", "jangan", "coba", "minta", "sampai", "pagi", "siang",
        "malam", "besok", "lusa", "sekarang", "berapa", "satu", "dua", "tiga", "empat",
        "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh", "coba", "gimana",
        "kalo", "tau", "nggak", "gak", "nih", "dong", "deh", "sih", "banget", "bener",
        "kalo", "dulu", "nanti", "apa", "aja", "ya", "lah", "loh", "kan", "pun",
        "fyi", "gue", "lu", "lo", "cuy", "abis", "emang", "tuh", "gitu", "gini",
    }
    ind_slang = {
        "gue", "lu", "lo", "cuy", "nih", "dong", "deh", "sih", "banget", "abis", "gitu",
        "gini", "tuh", "emang", "ya", "lah", "loh", "kan", "ngeh", "bentar", "baca",
    }
    en_words = {
        "i", "you", "we", "they", "he", "she", "it", "am", "is", "are", "was", "were",
        "be", "been", "being", "want", "need", "can", "could", "would", "should", "will",
        "do", "does", "did", "have", "has", "had", "to", "for", "from", "with", "this",
        "that", "these", "those", "please", "book", "flight", "schedule", "jeddah", "mingo",
        "japan", "airport", "hotel", "morning", "afternoon", "night", "tomorrow", "sunday",
        "january", "google", "flights", "skyscanner", "help", "thanks", "thank", "yes", "no",
        "go", "check", "directly", "real", "time", "can", "you", "me", "my", "your", "our",
        "hello", "hi", "hey", "okay", "ok", "please", "directly", "book", "route", "ticket",
    }
    en_slang = {
        "yo", "dude", "bro", "sup", "pls", "lol", "idk", "u", "ur", "gonna", "wanna",
        "kinda", "ain", "aint", "nah", "hmm", "okay", "ok",
    }
    ar_words = {
        "السلام", "عليكم", "وعليكم", "صل", "سلم", "سلام", "الله", "الحمد", "لله",
        "الحمد لله", "سبحان", "تعالى", "يا", "ربي", "ربي", "ربنا", "تبارك", "الله",
        "مساء", "صباح", "مرحبا", "كيف", "الحال", "اليوم", "احسن", "شكرا", "برك",
        "شكرا", "الحمدلله", "اللهم", "تسلمي", "فيه", "عربي", "عربيه",
    }
    ar_slang = {
        "assalamualaikum", "alhamdulillah", "inshaallah", "masyaallah", "subhanallah",
        "habibi", "habibti", "wallahi", "jazakallah", "jazakillah", "salam", "salamualaikum",
    }

    ind_count = 0
    en_count = 0
    ar_count = 0

    for token in tokens:
        if re.fullmatch(r"[ء-ي]+", token):
            ar_count += 1
            continue

        if token in ar_words or token in ar_slang:
            ar_count += 1
            continue

        if token in en_words or token in en_slang:
            en_count += 1
            continue

        if token in ind_words or token in ind_slang or token.endswith(("nya", "lah", "kan", "kah", "pun", "ku", "mu", "deh", "sih", "dong", "nih")):
            ind_count += 1
            continue

        if token.endswith(("ing", "tion", "ment", "ly", "ize", "ise", "ed", "er")):
            en_count += 1
            continue

        ind_count += 1

    total = len(tokens)
    ratios = {}
    if ar_count:
        ratios["AR"] = round(ar_count / total, 3)
    if en_count:
        ratios["EN"] = round(en_count / total, 3)
    if ind_count:
        ratios["IND"] = round(ind_count / total, 3)

    if "EN" in ratios and ratios["EN"] >= 0.8 and ratios.get("IND", 0) < 0.2:
        ratios = {"EN": round(ratios["EN"], 3)}
    elif "AR" in ratios and ratios["AR"] >= 0.8 and ratios.get("IND", 0) < 0.2:
        ratios = {"AR": round(ratios["AR"], 3)}

    tags = {key: f"{value:.0%}" for key, value in ratios.items()}
    return tags, ratios


def _build_json_response(
    transcript: str,
    llm_response: str,
    normalized_text: str,
    mode: str,
    output_audio_path: str,
    background: BackgroundTask | None = None,
):
    tags, ratios = _analyze_language(transcript)
    with open(output_audio_path, "rb") as audio_file:
        audio_b64 = base64.b64encode(audio_file.read()).decode("utf-8")

    response = JSONResponse(
        {
            "status": "success",
            "session_id": uuid.uuid4().hex,
            "mode": mode,
            "user_text": transcript,
            "transcription": transcript,
            "normalized_text": normalized_text,
            "language_tags": tags,
            "language_ratios": ratios,
            "llm_response": llm_response,
            "response_text": llm_response,
            "audio_base64": audio_b64,
        }
    )

    if background is not None:
        response.background = background

    return response


@app.get("/")
def root():
    return {"message": "Backend aktif. Gunakan endpoint POST /app atau /voice-chat."}


@app.post("/voice-chat")
@app.post("/app")
async def voice_chat(
    file: UploadFile = File(...),
    mode: str = "preserve",
    format: str = "file",
):
    ext = get_file_ext(file.filename)
    upload_path = os.path.join(TEMP_DIR, f"upload_{uuid.uuid4()}{ext}")
    output_audio_path = None

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="File audio kosong.")

        with open(upload_path, "wb") as temp_file:
            temp_file.write(file_bytes)

        transcript = transcribe_audio_file(upload_path)
        if transcript.startswith("[ERROR]"):
            raise HTTPException(status_code=500, detail=transcript)

        normalized_text = normalize_text(transcript)
        prompt_text = normalized_text if mode == "normalized" else transcript
        llm_response = generate_response(prompt_text)
        if llm_response.startswith("[ERROR]"):
            raise HTTPException(status_code=500, detail=llm_response)

        output_audio_path = transcribe_text_to_speech(llm_response)
        cleanup_task = BackgroundTask(lambda: [safe_delete(upload_path), safe_delete(output_audio_path)])

        if format == "json":
            return _build_json_response(
                transcript=transcript,
                llm_response=llm_response,
                normalized_text=normalized_text,
                mode=mode,
                output_audio_path=output_audio_path,
                background=cleanup_task,
            )

        return FileResponse(
            output_audio_path,
            media_type="audio/wav",
            filename="chatbot_response.wav",
            headers={
                "X-Transcript": transcript.encode("ascii", "ignore").decode(),
                "X-LLM-Response": llm_response.encode("ascii", "ignore").decode(),
            },
            background=cleanup_task,
        )
    except HTTPException:
        safe_delete(upload_path)
        safe_delete(output_audio_path)
        raise
    except Exception as exc:
        safe_delete(upload_path)
        safe_delete(output_audio_path)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/debug-text")
async def debug_text(file: UploadFile = File(...)):
    """Endpoint bantuan untuk melihat hasil STT dan LLM tanpa TTS."""
    ext = get_file_ext(file.filename)
    upload_path = os.path.join(TEMP_DIR, f"debug_{uuid.uuid4()}{ext}")
    try:
        with open(upload_path, "wb") as temp_file:
            temp_file.write(await file.read())
        transcript = transcribe_audio_file(upload_path)
        response = generate_response(transcript) if not transcript.startswith("[ERROR]") else ""
        return JSONResponse({"transcript": transcript, "response": response})
    finally:
        safe_delete(upload_path)
