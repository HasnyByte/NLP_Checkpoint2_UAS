import base64
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import gradio as gr
import pandas as pd
import requests
import scipy.io.wavfile


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/voice-chat")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DATA_DIR = PROJECT_ROOT / "data" / "audio"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BATCH_LIMIT_CHOICES = [10, 50, 100, 250, 500]
BATCH_TABLE_HEADERS = [
    "No",
    "Nama File",
    "Hasil STT",
    "Respons LLM",
    "Status",
    "Waktu Proses / Latency",
]
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


# ---------------------------------------------------------------------------
# Helper: markup builders
# ---------------------------------------------------------------------------
def status_markup(state="idle", message="Menunggu input suara."):
    state_class = {
        "idle": "idle", "processing": "processing",
        "success": "success", "warning": "warning", "error": "error"
    }.get(state, "idle")
    return f"""
    <div class="status-box {state_class}">
        <span class="status-dot"></span>
        <span class="status-msg">{message}</span>
    </div>
    """


def language_tags_markup(tags):
    if not tags:
        return "<div class='empty-note'>Belum ada ujaran yang diproses.</div>"
    if isinstance(tags, str):
        return f"<div class='tag-cloud'>{tags}</div>"
    chips = []
    items = tags.items() if isinstance(tags, dict) else enumerate(tags)
    for key, value in items:
        chips.append(f"<span class='lang-chip'><b>{key}</b>{value}</span>")
    return f"<div class='tag-cloud'>{''.join(chips)}</div>"


def ratio_text(ratios):
    if not ratios:
        return ""
    if isinstance(ratios, str):
        return ratios
    color_map = {"IND": "IND", "ID": "IND", "EN": "EN", "AR": "AR", "ID-Slang": "Slang"}
    return "  |  ".join(f"{color_map.get(lang, lang)}: {ratio}" for lang, ratio in ratios.items())


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------
def _write_temp_wav(sample_rate, audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sample_rate, audio_data)
        return tmpfile.name


def _save_response_audio(response):
    path = os.path.join(tempfile.gettempdir(), f"tts_output_{uuid.uuid4()}.wav")
    with open(path, "wb") as f:
        f.write(response.content)
    return path


def _save_base64_audio(audio_base64, session_id):
    path = os.path.join(tempfile.gettempdir(), f"gradio_res_{session_id}.wav")
    with open(path, "wb") as f:
        f.write(base64.b64decode(audio_base64))
    return path


# ---------------------------------------------------------------------------
# Batch pipeline analysis utilities
# ---------------------------------------------------------------------------
def summary_markup(
    total=0,
    success=0,
    failed=0,
    empty_stt=0,
    empty_llm=0,
    avg_latency=0.0,
    note="Belum ada analisis yang dijalankan.",
):
    return f"""
    <div class="summary-grid">
        <div class="summary-item"><span>Total Audio</span><b>{total}</b></div>
        <div class="summary-item success"><span>Sukses</span><b>{success}</b></div>
        <div class="summary-item danger"><span>Gagal</span><b>{failed}</b></div>
        <div class="summary-item warning"><span>STT Kosong</span><b>{empty_stt}</b></div>
        <div class="summary-item warning"><span>LLM Kosong</span><b>{empty_llm}</b></div>
        <div class="summary-item"><span>Rata-rata Latency</span><b>{avg_latency:.2f} detik</b></div>
    </div>
    <div class="summary-note">{note}</div>
    """


def _empty_batch_result(message, state="idle"):
    return (
        pd.DataFrame(columns=BATCH_TABLE_HEADERS),
        status_markup(state, message),
        summary_markup(note=message),
    )


def _list_audio_files(audio_dir=AUDIO_DATA_DIR):
    if not audio_dir.exists():
        return []

    files = [
        path
        for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    return sorted(files, key=lambda path: path.name.lower())


def _truncate_text(text, max_chars=260):
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _is_blank_or_error(text):
    text = str(text or "").strip()
    return not text or text.startswith("[ERROR]")


def _clean_error_message(message):
    raw_message = str(message or "").strip()
    if not raw_message:
        return "Error tidak diketahui"

    if "Teks kosong" in raw_message or "teks kosong" in raw_message:
        return "TTS gagal - teks kosong"

    if raw_message.startswith("[ERROR]"):
        raw_message = raw_message.replace("[ERROR]", "", 1).strip()

    try:
        parsed = json.loads(raw_message)
        raw_message = str(parsed.get("detail") or parsed.get("message") or raw_message)
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    lowered = raw_message.lower()
    if "whisper timeout" in lowered:
        return "STT timeout"
    if "whisper" in lowered:
        return "STT gagal"
    if "gemini" in lowered or "llm" in lowered:
        return "LLM gagal"
    if "timeout" in lowered:
        return "Request timeout"
    if "connection" in lowered or "tersambung" in lowered:
        return "Backend tidak tersambung"

    return _truncate_text(raw_message, 120)


def _process_audio_file_for_batch(file_path, mode):
    start_time = time.time()

    try:
        from app.llm import generate_response
        from app.stt import transcribe_audio_file
        from app.utils import normalize_text

        stt_text = transcribe_audio_file(str(file_path))
        latency = round(time.time() - start_time, 2)

        if _is_blank_or_error(stt_text):
            status = (
                "STT kosong - audio tidak terbaca jelas"
                if not str(stt_text or "").strip()
                else _clean_error_message(stt_text)
            )
            return "", "", status, latency

        prompt_text = normalize_text(stt_text) if mode == "normalized" else stt_text
        llm_response = generate_response(prompt_text)
        latency = round(time.time() - start_time, 2)

        if _is_blank_or_error(llm_response):
            status = (
                "Respons LLM kosong"
                if not str(llm_response or "").strip()
                else _clean_error_message(llm_response)
            )
            return stt_text, "", status, latency

        if "teks kosong" in str(llm_response).lower():
            return stt_text, "", "TTS gagal - teks kosong", latency

        return stt_text, llm_response, "Sukses", latency

    except requests.exceptions.Timeout:
        return "", "", "Error: request timeout.", round(time.time() - start_time, 2)
    except requests.exceptions.ConnectionError:
        return "", "", "Error: backend FastAPI tidak tersambung.", round(time.time() - start_time, 2)
    except Exception as exc:
        return "", "", _clean_error_message(exc), round(time.time() - start_time, 2)


def run_batch_pipeline_analysis(limit, mode):
    if limit is None:
        return _empty_batch_result(
            "Pilih jumlah data terlebih dahulu sebelum menjalankan Analisis Pipeline.",
            "warning",
        )

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return _empty_batch_result("Jumlah data tidak valid.", "error")

    if limit not in BATCH_LIMIT_CHOICES or limit > 500:
        return _empty_batch_result("Batas maksimal analisis adalah 500 data.", "error")

    if not AUDIO_DATA_DIR.exists():
        return _empty_batch_result(
            f"Folder data/audio tidak ditemukan di: {AUDIO_DATA_DIR}",
            "error",
        )

    audio_files = _list_audio_files()
    if not audio_files:
        return _empty_batch_result("Folder data/audio kosong atau tidak berisi file audio yang didukung.", "warning")

    selected_files = audio_files[:limit]
    rows = []

    for index, file_path in enumerate(selected_files, start=1):
        stt_text, llm_response, status, latency = _process_audio_file_for_batch(file_path, mode)
        rows.append(
            {
                "No": index,
                "Nama File": _truncate_text(file_path.name, 90),
                "Hasil STT": _truncate_text(stt_text, 260),
                "Respons LLM": _truncate_text(llm_response, 260),
                "Status": _truncate_text(status, 150),
                "Waktu Proses / Latency": f"{latency:.2f} detik",
                "_latency": latency,
            }
        )

    success_count = sum(1 for row in rows if row["Status"] == "Sukses")
    failed_count = len(rows) - success_count
    empty_stt_count = sum(1 for row in rows if "STT kosong" in row["Status"])
    empty_llm_count = sum(1 for row in rows if "Respons LLM kosong" in row["Status"])
    avg_latency = sum(row["_latency"] for row in rows) / len(rows) if rows else 0.0

    table_df = pd.DataFrame(rows).drop(columns=["_latency"])
    summary = (
        f"Analisis selesai. Diproses {len(rows)} dari {len(audio_files)} file audio. "
        f"Sukses: {success_count}. Gagal: {failed_count}. "
        f"STT kosong: {empty_stt_count}. Respons LLM kosong: {empty_llm_count}. "
        f"Rata-rata latency: {avg_latency:.2f} detik."
    )
    state = "success" if failed_count == 0 else "warning"
    return (
        table_df,
        status_markup(state, summary),
        summary_markup(
            total=len(rows),
            success=success_count,
            failed=failed_count,
            empty_stt=empty_stt_count,
            empty_llm=empty_llm_count,
            avg_latency=avg_latency,
            note=summary,
        ),
    )


# ---------------------------------------------------------------------------
# Pipeline (logic unchanged)
# ---------------------------------------------------------------------------
def voice_chat_pipeline(audio, mode):
    if audio is None:
        return (
            None, "", "",
            language_tags_markup(None), "", "",
            status_markup("idle", "Silakan rekam suara terlebih dahulu."),
            "Belum ada audio. Rekam suara, lalu tekan Proses Pipeline.",
        )

    sample_rate, audio_data = audio
    input_audio_path = _write_temp_wav(sample_rate, audio_data)

    try:
        with open(input_audio_path, "rb") as audio_file:
            files = {"file": ("voice.wav", audio_file, "audio/wav")}
            data = {"mode": mode}
            response = requests.post(
                f"{BACKEND_URL}?format=json", files=files, data=data, timeout=120
            )

        if response.status_code != 200:
            return (
                None, "", "",
                language_tags_markup("<span class='error-text'>Backend mengembalikan error.</span>"),
                "", "",
                status_markup("error", f"Backend error HTTP {response.status_code}."),
                response.text,
            )

        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            output_audio_path = _save_response_audio(response)
            return (
                output_audio_path,
                "Backend mengirim audio langsung tanpa metadata JSON.",
                "-",
                language_tags_markup("Metadata tagging tidak tersedia pada respons audio langsung."),
                "-",
                "Audio balasan berhasil dibuat. Klik play pada Tahap 5 untuk mendengarkan.",
                status_markup("success", "Pipeline sukses. Audio balasan siap diputar."),
                "Audio siap. Gunakan player pada Tahap 5.",
            )

        result = response.json()
        if result.get("status") not in {None, "success"}:
            return (
                None, "", "",
                language_tags_markup("<span class='error-text'>Pipeline gagal diproses.</span>"),
                "", "",
                status_markup("error", result.get("message", "Pipeline gagal diproses.")),
                result.get("message", "Pipeline gagal diproses."),
            )

        session_id = result.get("session_id", uuid.uuid4().hex)
        output_audio_path = None
        if result.get("audio_base64"):
            output_audio_path = _save_base64_audio(result["audio_base64"], session_id)

        user_text       = result.get("user_text") or result.get("transcription") or ""
        normalized_text = result.get("normalized_text") or ""
        language_tags   = language_tags_markup(result.get("language_tags"))
        ratios          = ratio_text(result.get("language_ratios"))
        llm_response    = result.get("llm_response") or result.get("response_text") or ""

        return (
            output_audio_path,
            user_text, normalized_text, language_tags, ratios, llm_response,
            status_markup("success", f"Pipeline sukses. Session ID: {session_id}"),
            "Audio balasan siap diputar pada Tahap 5." if output_audio_path else "Metadata sukses, tetapi audio tidak ditemukan.",
        )

    except requests.exceptions.Timeout:
        return (
            None, "", "",
            language_tags_markup("<span class='error-text'>Request timeout.</span>"),
            "", "",
            status_markup("error", "Backend terlalu lama merespons."),
            "Coba gunakan rekaman yang lebih pendek.",
        )
    except requests.exceptions.ConnectionError:
        return (
            None, "", "",
            language_tags_markup("<span class='error-text'>Backend tidak tersambung.</span>"),
            "", "",
            status_markup("error", "Tidak bisa terhubung ke backend FastAPI."),
            "Pastikan backend berjalan di localhost:8000.",
        )
    except Exception as exc:
        return (
            None, "", "",
            language_tags_markup("<span class='error-text'>Terjadi kesalahan.</span>"),
            "", "",
            status_markup("error", "Terjadi kesalahan saat menjalankan pipeline."),
            str(exc),
        )
    finally:
        if os.path.exists(input_audio_path):
            os.remove(input_audio_path)


# ---------------------------------------------------------------------------
# Gradio theme — light, matching our palette
# ---------------------------------------------------------------------------
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "Inter", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Plus Jakarta Sans"), "monospace"],
).set(
    # body / page
    body_background_fill="#f0f4ff",
    body_background_fill_dark="#f0f4ff",
    body_text_color="#0f172a",
    body_text_color_dark="#0f172a",
    body_text_color_subdued="#5b6f8e",
    body_text_color_subdued_dark="#5b6f8e",
    # input / textbox
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_background_fill_focus="#ffffff",
    input_background_fill_focus_dark="#ffffff",
    input_border_color="#dce5f5",
    input_border_color_dark="#dce5f5",
    input_border_color_focus="#4f46e5",
    input_border_color_focus_dark="#4f46e5",
    input_placeholder_color="#94a3b8",
    input_placeholder_color_dark="#94a3b8",
    # block
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#dce5f5",
    block_border_color_dark="#dce5f5",
    block_label_text_color="#5b6f8e",
    block_label_text_color_dark="#5b6f8e",
    block_title_text_color="#0f172a",
    block_title_text_color_dark="#0f172a",
    block_shadow="0 4px 24px rgba(15,23,80,0.07)",
    block_radius="20px",
    # panel
    panel_background_fill="#f7f9ff",
    panel_background_fill_dark="#f7f9ff",
    # button primary
    button_primary_background_fill="linear-gradient(135deg,#4f46e5,#06b6d4)",
    button_primary_background_fill_dark="linear-gradient(135deg,#4f46e5,#06b6d4)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_border_color="transparent",
    button_primary_border_color_dark="transparent",
    # button secondary
    button_secondary_background_fill="#f7f9ff",
    button_secondary_background_fill_dark="#f7f9ff",
    button_secondary_text_color="#4f46e5",
    button_secondary_text_color_dark="#4f46e5",
    button_secondary_border_color="#dce5f5",
    button_secondary_border_color_dark="#dce5f5",
    # radio / checkbox
    checkbox_background_color="#ffffff",
    checkbox_background_color_dark="#ffffff",
    checkbox_border_color="#dce5f5",
    checkbox_border_color_dark="#dce5f5",
    checkbox_label_background_fill="#f7f9ff",
    checkbox_label_background_fill_dark="#f7f9ff",
    checkbox_label_text_color="#0f172a",
    checkbox_label_text_color_dark="#0f172a",
    # border radius
    input_radius="13px",
    button_large_radius="999px",
    button_small_radius="999px",
    # color accent
    color_accent="#4f46e5",
    color_accent_soft="#edf2ff",
    color_accent_soft_dark="#edf2ff",
    # error
    error_background_fill="#fff5f5",
    error_border_color="#fecaca",
    error_text_color="#b91c1c",
)


# ---------------------------------------------------------------------------
# CSS overrides
# ---------------------------------------------------------------------------
css = """
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Outfit:wght@300;400;600;700;900&display=swap');

/* ── Tokens ── */
:root {
  --bg:            #f0f4ff;
  --surface:       #ffffff;
  --surface-soft:  #f7f9ff;
  --surface-muted: #edf2ff;
  --text:          #0f172a;
  --muted:         #5b6f8e;
  --line:          #dce5f5;
  --brand:         #4f46e5;
  --brand-2:       #06b6d4;
  --brand-3:       #8b5cf6;
  --error:         #ef4444;
  --success:       #10b981;
  --shadow:        0 20px 60px rgba(15,23,80,0.10);
  --shadow-soft:   0 8px 28px rgba(15,23,80,0.07);
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── Force light background everywhere in Gradio ── */
html, body {
  background: var(--bg) !important;
  color: var(--text) !important;
}

.gradio-container,
.gradio-container > .main,
.gradio-container .wrap {
  background: transparent !important;
  font-family: "Plus Jakarta Sans", Inter, sans-serif !important;
  color: var(--text) !important;
}

.gradio-container {
  min-height: 100vh;
  padding: 0 16px 56px !important;
  background:
    radial-gradient(ellipse at 8% 0%,  rgba(79,70,229,0.12)  0%, transparent 48%),
    radial-gradient(ellipse at 92% 4%, rgba(6,182,212,0.10)  0%, transparent 44%),
    var(--bg) !important;
}

/* Kill default Gradio borders on top-level container */
.block, .form { border: none !important; box-shadow: none !important; background: transparent !important; }

footer, .footer { display: none !important; }
.contain { max-width: none !important; }

/* Make all Gradio textboxes light */
textarea, input[type="text"], input[type="number"] {
  background: #ffffff !important;
  color: #0f172a !important;
  border-color: var(--line) !important;
}

label span, .svelte-1f354aw { color: var(--muted) !important; }

/* ── App shell ── */
.app-shell {
  width: min(1200px, 100%);
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 0;
}

/* ════════════════════════════════════════════
   HERO — redesigned
   ════════════════════════════════════════════ */
.hero {
  position: relative;
  overflow: hidden;
  padding: 44px 48px 40px;
  background:
    radial-gradient(ellipse at 0% 100%, rgba(139,92,246,0.18) 0%, transparent 55%),
    radial-gradient(ellipse at 100% 0%,  rgba(6,182,212,0.16)  0%, transparent 50%),
    linear-gradient(160deg, #ffffff 0%, #f0f4ff 100%);
  border-bottom: 1px solid var(--line);
  margin-bottom: 28px;
  border-radius: 28px 28px 0 0;
}

/* decorative blobs */
.hero::before,
.hero::after {
  content: "";
  position: absolute;
  border-radius: 50%;
  pointer-events: none;
}
.hero::before {
  width: 420px; height: 420px;
  top: -160px; right: -100px;
  background: radial-gradient(circle, rgba(79,70,229,0.10) 0%, transparent 70%);
}
.hero::after {
  width: 280px; height: 280px;
  bottom: -120px; left: 30%;
  background: radial-gradient(circle, rgba(6,182,212,0.09) 0%, transparent 70%);
}

.hero-inner {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: center;
  gap: 32px;
  max-width: 1200px;
  margin: 0 auto;
}

/* eyebrow label */
.hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 14px;
  border-radius: 999px;
  border: 1px solid rgba(79,70,229,0.25);
  background: rgba(79,70,229,0.07);
  color: var(--brand);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 18px;
}

.hero-eyebrow::before {
  content: "";
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--brand);
}

/* headline — two fonts for visual contrast */
.hero-title {
  margin: 0 0 6px;
  font-family: "Outfit", "Plus Jakarta Sans", sans-serif;
  font-size: clamp(28px, 3.8vw, 52px);
  font-weight: 900;
  line-height: 1.05;
  letter-spacing: -0.02em;
}

.hero-title .word-main {
  background: linear-gradient(100deg, #4f46e5 10%, #06b6d4 80%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-title .word-sub {
  display: block;
  font-size: 0.55em;
  font-weight: 700;
  font-family: "Plus Jakarta Sans", sans-serif;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted);
  -webkit-text-fill-color: var(--muted);
  margin-bottom: 8px;
}

.hero-desc {
  margin: 14px 0 0;
  color: var(--muted);
  font-size: 15px;
  line-height: 1.7;
  font-weight: 500;
  max-width: 560px;
}

/* pipeline badge row */
.hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 22px;
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: 10px;
  background: rgba(255,255,255,0.85);
  border: 1px solid var(--line);
  color: var(--text);
  font-size: 11.5px;
  font-weight: 700;
  box-shadow: 0 2px 8px rgba(15,23,80,0.05);
}

.hero-badge-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
}

/* visual panel on the right */
.hero-visual {
  position: relative;
  width: 160px;
  height: 160px;
  flex-shrink: 0;
}

.hero-visual-ring {
  position: absolute;
  inset: 0;
  border-radius: 50%;
  border: 1.5px dashed rgba(79,70,229,0.22);
  animation: spin-slow 18s linear infinite;
}

.hero-visual-ring-2 {
  position: absolute;
  inset: 16px;
  border-radius: 50%;
  border: 1.5px dashed rgba(6,182,212,0.22);
  animation: spin-slow 12s linear infinite reverse;
}

@keyframes spin-slow {
  to { transform: rotate(360deg); }
}

.hero-visual-core {
  position: absolute;
  inset: 30px;
  border-radius: 30px;
  background: linear-gradient(135deg, #4f46e5, #06b6d4);
  display: grid;
  place-items: center;
  box-shadow: 0 18px 44px rgba(79,70,229,0.30);
}

.hero-visual-core svg {
  width: 42px; height: 42px;
  fill: none;
  stroke: #ffffff;
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}

/* dots decoration */
.hero-dots {
  position: absolute;
  top: -8px; right: -8px;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 5px;
}

.hero-dots span {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: rgba(79,70,229,0.25);
}

/* ════════════════════════════════════════════
   LAYOUT
   ════════════════════════════════════════════ */
.layout {
  display: grid !important;
  grid-template-columns: minmax(300px, 42%) minmax(0, 1fr);
  gap: 18px;
  align-items: start;
}

/* ════════════════════════════════════════════
   CARD
   ════════════════════════════════════════════ */
.card {
  border: 1px solid var(--line) !important;
  border-radius: 24px !important;
  background: #ffffff !important;
  box-shadow: var(--shadow) !important;
  padding: 26px !important;
  overflow: visible;
}

/* ── Panel header ── */
.panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 16px;
}

.panel-title {
  margin: 0;
  color: var(--text);
  font-size: 17px;
  font-weight: 900;
}

.panel-copy {
  margin: 5px 0 0;
  color: var(--muted);
  font-size: 12.5px;
  line-height: 1.55;
  font-weight: 600;
}

.chip {
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 0 10px;
  height: 24px;
  border-radius: 999px;
  border: 1px solid rgba(79,70,229,0.25);
  background: rgba(79,70,229,0.07);
  color: var(--brand);
  font-size: 10.5px;
  font-weight: 800;
  white-space: nowrap;
}
.chip::before {
  content: "";
  width: 6px; height: 6px;
  border-radius: 50%;
  background: currentColor;
}

/* ── Voice visual card ── */
.voice-card {
  padding: 12px;
  border-radius: 16px;
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(6,182,212,0.07) 0%, rgba(79,70,229,0.04) 100%);
  margin-bottom: 12px;
}

.voice-visual {
  height: 104px;
  display: grid;
  place-items: center;
}

.mic-disc {
  width: 72px; height: 72px;
  border-radius: 22px;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, #4f46e5, #06b6d4);
  box-shadow: 0 14px 36px rgba(79,70,229,0.28);
}

.mic-disc svg {
  width: 31px; height: 31px;
  fill: none;
  stroke: #fff;
  stroke-width: 1.9;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.audio-field-label {
  margin: 0 0 8px 2px;
  color: var(--text);
  font-size: 12.5px;
  font-weight: 900;
}

/* ── Audio input/output ── */
#audio-input, #audio-output {
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  background: #ffffff !important;
  overflow: hidden !important;
  box-shadow: none !important;
}

#audio-input {
  min-height: 174px !important;
  padding: 10px !important;
}

#audio-input [role="tablist"] {
  display: flex !important;
  justify-content: center !important;
  gap: 6px !important;
  padding: 2px 2px 8px !important;
  margin-bottom: 6px !important;
  border-bottom: 1px solid var(--line) !important;
}

#audio-input [role="tabpanel"],
#audio-input [data-testid="audio"],
#audio-input [data-testid="file-upload"],
#audio-input .upload-container,
#audio-input .dropzone,
#audio-input .waveform-container {
  width: 100% !important;
  min-height: 108px !important;
  border-radius: 13px !important;
  background: var(--surface-soft) !important;
}

#audio-input [data-testid="file-upload"],
#audio-input .upload-container,
#audio-input .dropzone {
  display: grid !important;
  place-items: center !important;
  padding: 14px !important;
  border: 1px dashed rgba(79,70,229,0.28) !important;
}

#audio-input audio,
#audio-output audio {
  width: 100% !important;
  min-height: 42px !important;
}

#audio-input .file-preview,
#audio-input [data-testid="file-preview"] {
  width: 100% !important;
  border-radius: 12px !important;
  padding: 10px 12px !important;
  background: #ffffff !important;
}

/* Force all inner wrappers white */
#audio-input > *, #audio-input div,
#audio-output > *, #audio-output div {
  background: #ffffff !important;
  color: #0f172a !important;
}

/* Kill ALL borders inside audio components globally */
#audio-input *, #audio-output * {
  border-color: var(--line) !important;
  outline: none !important;
}

/* Kill black label/block header border ("Rekam ujaran Anda" / "Balasan Suara Asisten") */
#audio-input .block,
#audio-input .label-wrap,
#audio-output .block,
#audio-output .label-wrap {
  border: none !important;
  border-bottom: none !important;
  box-shadow: none !important;
  background: transparent !important;
}

/* Remove black border/outline on Gradio audio tab bar and internal wrappers */
#audio-input .tab-nav,
#audio-input .tabs,
#audio-input .tabitem,
#audio-input [role="tablist"],
#audio-input [role="tab"],
#audio-input .svelte-tab-bar,
#audio-input .wrap,
#audio-input > div > div,
#audio-output .tab-nav,
#audio-output .tabs,
#audio-output .tabitem,
#audio-output [role="tablist"],
#audio-output .wrap,
#audio-output > div > div {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  background: #ffffff !important;
}

/* Selected tab — remove black bottom border */
#audio-input [role="tab"][aria-selected="true"],
#audio-input [role="tab"].selected,
#audio-input button[role="tab"] {
  border: none !important;
  border-bottom: 2px solid var(--brand) !important;
  outline: none !important;
  box-shadow: none !important;
  color: var(--brand) !important;
  font-family: "Plus Jakarta Sans", Inter, sans-serif !important;
  font-size: 12.5px !important;
  font-weight: 700 !important;
}

/* Unselected tab */
#audio-input [role="tab"]:not([aria-selected="true"]) {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  color: var(--muted) !important;
  font-family: "Plus Jakarta Sans", Inter, sans-serif !important;
  font-size: 12.5px !important;
  font-weight: 700 !important;
}

/* Fix "No microphone found" and all status/helper text inside audio component */
#audio-input .no-mic,
#audio-input .mic-error,
#audio-input [class*="no-mic"],
#audio-input [class*="error"],
#audio-input p,
#audio-input span:not([role="tab"]),
#audio-input .message,
#audio-input .status-message,
#audio-input .sr-only,
#audio-input small {
  font-family: "Plus Jakarta Sans", Inter, sans-serif !important;
  font-size: 10.5px !important;
  font-weight: 700 !important;
  line-height: 1.35 !important;
  color: var(--muted) !important;
}

#audio-input label, #audio-output label {
  color: var(--muted) !important;
  font-weight: 700 !important;
  font-size: 12.5px !important;
  background: transparent !important;
}

#audio-input button, #audio-output button,
#audio-input [role="button"], #audio-output [role="button"] {
  height: 34px !important;
  min-height: 34px !important;
  width: auto !important;
  min-width: 34px !important;
  padding: 0 11px !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  background: #f7f9ff !important;
  color: #0f172a !important;
  font-size: 12.5px !important;
  font-weight: 700 !important;
  font-family: "Plus Jakarta Sans", Inter, sans-serif !important;
  box-shadow: none !important;
}

/* Override: tab buttons shouldn't look like regular buttons */
#audio-input [role="tab"] {
  height: 32px !important;
  min-height: 32px !important;
  min-width: 108px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 0 14px !important;
  border-radius: 999px !important;
  background: var(--surface-soft) !important;
  border: 1px solid var(--line) !important;
  border-bottom: 1px solid var(--line) !important;
}

#audio-input [role="tab"][aria-selected="true"] {
  border-color: rgba(79,70,229,0.32) !important;
  background: #edf2ff !important;
}

#audio-input button:hover, #audio-output button:hover,
#audio-input [role="button"]:hover, #audio-output [role="button"]:hover {
  background: #edf2ff !important;
  border-color: rgba(79,70,229,0.3) !important;
}

#audio-input button[aria-label="Drop an audio file here to upload"] {
  position: relative !important;
  width: 100% !important;
  height: 108px !important;
  min-height: 108px !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 6px !important;
  padding: 14px !important;
  border: 1px dashed rgba(79,70,229,0.28) !important;
  border-radius: 14px !important;
  background: var(--surface-soft) !important;
  color: var(--text) !important;
  text-align: center !important;
  overflow: hidden !important;
}

#audio-input button[aria-label="Drop an audio file here to upload"] span,
#audio-input button[aria-label="Drop an audio file here to upload"] p,
#audio-input button[aria-label="Drop an audio file here to upload"] div {
  background: transparent !important;
  color: var(--muted) !important;
  font-size: 11.5px !important;
  font-weight: 700 !important;
  line-height: 1.4 !important;
}

#audio-input button[aria-label="Drop an audio file here to upload"]:hover {
  background: #edf2ff !important;
  border-color: rgba(79,70,229,0.45) !important;
}

#audio-input input[type="file"] {
  position: absolute !important;
  inset: 0 !important;
  width: 100% !important;
  height: 100% !important;
  opacity: 0 !important;
  cursor: pointer !important;
}

/* Shrink the X (clear) button */
#audio-input button[aria-label="Clear"],
#audio-input button[aria-label="clear"],
#audio-output button[aria-label="Clear"],
#audio-output button[aria-label="clear"] {
  height: 26px !important;
  min-height: 26px !important;
  width: 26px !important;
  min-width: 26px !important;
  padding: 0 !important;
  border-radius: 8px !important;
  display: grid !important;
  place-items: center !important;
}

#audio-input button[aria-label="Clear"] svg,
#audio-input button[aria-label="clear"] svg,
#audio-output button[aria-label="Clear"] svg,
#audio-output button[aria-label="clear"] svg {
  width: 11px !important;
  height: 11px !important;
}

#audio-input svg, #audio-output svg {
  width: 15px !important;
  height: 15px !important;
  color: #5b6f8e !important;
  stroke: #5b6f8e !important;
}

#audio-input [role="tablist"] {
  border-bottom: 1px solid var(--line) !important;
}

#audio-input [role="tabpanel"],
#audio-input [data-testid="audio"],
#audio-input [data-testid="file-upload"],
#audio-input .upload-container,
#audio-input .dropzone,
#audio-input .waveform-container {
  background: var(--surface-soft) !important;
}

#audio-input [data-testid="file-upload"],
#audio-input .upload-container,
#audio-input .dropzone {
  border: 1px dashed rgba(79,70,229,0.28) !important;
}

/* Compact final pass for Gradio audio internals */
#audio-input,
#audio-input .wrap,
#audio-input [role="tabpanel"],
#audio-input [data-testid="audio"],
#audio-input [data-testid="file-upload"],
#audio-input .upload-container,
#audio-input .dropzone,
#audio-input .waveform-container {
  max-height: 176px !important;
}

#audio-input [role="tabpanel"],
#audio-input [data-testid="audio"],
#audio-input [data-testid="file-upload"],
#audio-input .upload-container,
#audio-input .dropzone,
#audio-input .waveform-container,
#audio-input button[aria-label="Drop an audio file here to upload"] {
  min-height: 108px !important;
  background: var(--surface-soft) !important;
  border-radius: 13px !important;
}

#audio-input [aria-label="No microphone found"],
#audio-input [class*="no-mic"],
#audio-input [class*="NoMicrophone"],
#audio-input .no-mic,
#audio-input .mic-error,
#audio-input p,
#audio-input small {
  font-size: 10.5px !important;
  line-height: 1.35 !important;
  font-weight: 700 !important;
  color: var(--muted) !important;
  text-align: center !important;
}

#audio-input button[aria-label="Record audio"],
#audio-input button[aria-label="Upload file"],
#audio-input [role="tab"],
#audio-input button[title*="Record"],
#audio-input button[title*="Upload"] {
  height: 32px !important;
  min-height: 32px !important;
  min-width: 104px !important;
  border-radius: 999px !important;
}

#audio-input button:not([aria-label="Drop an audio file here to upload"]) svg {
  width: 14px !important;
  height: 14px !important;
}

/* Audio input polish: balanced initial and upload states */
#audio-input {
  min-height: 154px !important;
  max-height: 164px !important;
  padding: 10px 12px !important;
  border-radius: 16px !important;
}

#audio-input [role="tablist"] {
  justify-content: center !important;
  gap: 10px !important;
  margin-bottom: 8px !important;
  padding-bottom: 8px !important;
}

#audio-input [role="tab"],
#audio-input button[aria-label="Record audio"],
#audio-input button[aria-label="Upload file"] {
  width: 126px !important;
  min-width: 126px !important;
  height: 34px !important;
  min-height: 34px !important;
  color: #0f172a !important;
  font-size: 12px !important;
  font-weight: 900 !important;
  gap: 7px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
}

#audio-input [role="tab"] svg,
#audio-input button[aria-label="Record audio"] svg,
#audio-input button[aria-label="Upload file"] svg {
  width: 13px !important;
  height: 13px !important;
}

#audio-input button[aria-label="Upload file"]::after {
  content: "Upload";
  color: #0f172a;
  font-size: 11.5px;
  font-weight: 900;
}

#audio-input button[aria-label="Record audio"]::after {
  content: "Mic";
  color: #0f172a;
  font-size: 11.5px;
  font-weight: 900;
}

#audio-input [role="tabpanel"],
#audio-input [data-testid="audio"],
#audio-input [data-testid="file-upload"],
#audio-input .upload-container,
#audio-input .dropzone,
#audio-input .waveform-container,
#audio-input button[aria-label="Drop an audio file here to upload"] {
  min-height: 92px !important;
  max-height: 96px !important;
  background: var(--surface-soft) !important;
}

#audio-input button[aria-label="Drop an audio file here to upload"] {
  height: 92px !important;
  padding: 10px 14px !important;
}

#audio-input button[aria-label="Drop an audio file here to upload"] span,
#audio-input button[aria-label="Drop an audio file here to upload"] p,
#audio-input button[aria-label="Drop an audio file here to upload"] div {
  color: #334155 !important;
  font-size: 11px !important;
  font-weight: 800 !important;
}

#audio-input [aria-label="No microphone found"],
#audio-input [class*="no-mic"],
#audio-input [class*="NoMicrophone"],
#audio-input .no-mic,
#audio-input .mic-error,
#audio-input p,
#audio-input small,
#audio-input option {
  color: #64748b !important;
  font-size: 10px !important;
  font-weight: 700 !important;
}

#audio-input button[aria-label="Clear"],
#audio-input button[aria-label="clear"] {
  opacity: 0.58 !important;
  transform: scale(0.9) !important;
  background: #ffffff !important;
  border-color: #e8eef9 !important;
}

#audio-input button[aria-label="Clear"]:hover,
#audio-input button[aria-label="clear"]:hover {
  opacity: 0.95 !important;
  background: #f8fbff !important;
}

/* ── Mode radio ── */
.mode-panel { margin-top: 14px; }

#mode-select {
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  background: var(--surface-soft) !important;
  padding: 12px 14px !important;
  box-shadow: none !important;
}
#mode-select label, #mode-select span {
  color: var(--text) !important;
  font-weight: 700 !important;
  font-size: 13px !important;
}
#mode-select .svelte-s1r2yt, #mode-select p {
  color: var(--muted) !important;
  font-size: 11.5px !important;
}

/* Radio item label — normal state */
#mode-select .wrap > label,
#mode-select [data-testid="radio-label"] {
  border-radius: 10px !important;
  border: 1px solid var(--line) !important;
  background: #ffffff !important;
  color: #0f172a !important;
  transition: background 0.18s, border-color 0.18s, color 0.18s !important;
}

/* Radio item label — hover */
#mode-select .wrap > label:hover,
#mode-select [data-testid="radio-label"]:hover {
  background: linear-gradient(135deg, #4f46e5, #06b6d4) !important;
  border-color: transparent !important;
  color: #ffffff !important;
}

/* Make the span text inside also turn white on hover */
#mode-select .wrap > label:hover span,
#mode-select [data-testid="radio-label"]:hover span {
  color: #ffffff !important;
}

/* Radio item label — selected */
#mode-select .wrap > label:has(input:checked),
#mode-select [data-testid="radio-label"]:has(input:checked) {
  background: linear-gradient(135deg, #4f46e5, #06b6d4) !important;
  border-color: transparent !important;
  color: #ffffff !important;
}

#mode-select .wrap > label:has(input:checked) span {
  color: #ffffff !important;
}

/* ── Submit button ── */
.send-zone {
  margin: 18px 0 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

#submit-button {
  width: 210px !important;
  max-width: 100% !important;
  height: 46px !important;
  min-height: 46px !important;
  border: 0 !important;
  border-radius: 999px !important;
  background: linear-gradient(135deg, #4f46e5, #06b6d4) !important;
  color: #ffffff !important;
  font-size: 13.5px !important;
  font-weight: 900 !important;
  letter-spacing: 0.01em !important;
  box-shadow: 0 10px 28px rgba(79,70,229,0.28) !important;
  transition: opacity 0.2s, transform 0.15s !important;
}
#submit-button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

.send-hint {
  color: var(--muted);
  font-size: 11.5px;
  font-weight: 700;
  text-align: center;
}

/* ── Help card ── */
.help-card {
  padding: 13px 15px;
  border-radius: 14px;
  border: 1px solid var(--line);
  background: var(--surface-soft);
  color: var(--muted);
  font-size: 12px;
  line-height: 1.6;
  font-weight: 600;
}

/* ── Flow cards (pipeline steps) ── */
.flow-card {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: var(--surface-soft);
  padding: 14px 16px;
  margin-bottom: 12px;
}
.flow-card:last-child { margin-bottom: 0; }

.step-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.step-number {
  width: 28px; height: 28px;
  border-radius: 9px;
  display: grid;
  place-items: center;
  flex-shrink: 0;
  color: #fff;
  background: linear-gradient(135deg, #4f46e5, #06b6d4);
  font-size: 11.5px;
  font-weight: 900;
}

.flow-title {
  margin: 0;
  color: var(--text);
  font-size: 13.5px;
  font-weight: 900;
}

.flow-note {
  margin: -4px 0 10px;
  color: var(--muted);
  font-size: 11.5px;
  line-height: 1.5;
  font-weight: 600;
}

/* Textbox inside flow-card */
.flow-card textarea, .flow-card input[type="text"] {
  color: #0f172a !important;
  background: #ffffff !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
}
.flow-card label span { color: var(--muted) !important; font-size: 12px !important; font-weight: 700 !important; }

/* ── Language tag cloud ── */
.tag-cloud {
  min-height: 38px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #ffffff;
  color: #0f172a;
  padding: 9px 11px;
  line-height: 1.7;
  word-break: break-word;
}

.lang-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  margin: 2px 3px;
  padding: 3px 9px;
  border-radius: 999px;
  background: rgba(6,182,212,0.09);
  color: #0f172a;
  border: 1px solid rgba(6,182,212,0.22);
  font-size: 12px;
  font-weight: 700;
}

.empty-note { color: var(--muted); font-style: italic; font-weight: 600; font-size: 12.5px; }
.error-text  { color: var(--error); font-weight: 800; }

/* ════════════════════════════════════════════
   STATUS BOX — always dark text
   ════════════════════════════════════════════ */
.status-box {
  display: flex;
  align-items: center;
  gap: 9px;
  min-height: 42px;
  padding: 10px 14px;
  border-radius: 14px;
  border: 1px solid var(--line);
  background: var(--surface-soft);
  margin-bottom: 12px;
}

.status-dot {
  width: 9px; height: 9px;
  border-radius: 50%;
  background: #94a3b8;
  flex-shrink: 0;
}

/* Text inside status box — always dark */
.status-msg {
  color: #0f172a !important;
  font-size: 13px;
  font-weight: 700;
}

.status-box.processing .status-dot { background: #14b8a6; }
.status-box.success    .status-dot { background: #10b981; }
.status-box.error      .status-dot { background: var(--error); }
.status-box.warning    .status-dot { background: #f59e0b; }

.status-box.error   .status-msg { color: #b91c1c !important; }
.status-box.success .status-msg { color: #065f46 !important; }
.status-box.warning .status-msg { color: #92400e !important; }

/* Status detail textbox */
#status-detail textarea {
  background: var(--surface-soft) !important;
  color: #0f172a !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  font-size: 12px !important;
  font-weight: 600 !important;
}
#status-detail label span { color: var(--muted) !important; font-size: 12px !important; font-weight: 700 !important; }

/* ── Batch analysis ── */
.analysis-card {
  margin-top: 18px;
}

.analysis-controls {
  display: flex !important;
  flex-direction: column !important;
  gap: 14px !important;
  width: min(760px, 100%) !important;
  margin: 0 0 18px !important;
  padding: 18px 20px !important;
  border: 1px solid var(--line) !important;
  border-radius: 20px !important;
  background: linear-gradient(180deg, #ffffff 0%, #f7f9ff 100%) !important;
  box-shadow: 0 12px 32px rgba(15,23,80,0.06) !important;
}

.analysis-control-copy h3 {
  margin: 0 0 5px;
  color: var(--text);
  font-size: 15px;
  font-weight: 900;
}

.analysis-control-copy p {
  margin: 0;
  max-width: 680px;
  color: var(--muted);
  font-size: 12.5px;
  line-height: 1.6;
  font-weight: 700;
}

.analysis-action-row {
  display: grid !important;
  grid-template-columns: minmax(220px, 1fr) minmax(200px, 0.82fr);
  gap: 14px !important;
  align-items: end !important;
  width: 100% !important;
}

#batch-limit {
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  background: #ffffff !important;
  padding: 0 !important;
  color: var(--text) !important;
  box-shadow: none !important;
  min-height: 46px !important;
  width: 100% !important;
}

#batch-limit *,
#batch-limit label,
#batch-limit input,
#batch-limit button,
#batch-limit [role="combobox"],
#batch-limit [role="listbox"],
#batch-limit [role="option"] {
  background: var(--surface-soft) !important;
  color: var(--text) !important;
  border-color: var(--line) !important;
  font-family: "Plus Jakarta Sans", Inter, sans-serif !important;
}

#batch-limit input,
#batch-limit [role="combobox"] {
  min-height: 38px !important;
  border-radius: 12px !important;
  background: #ffffff !important;
  color: var(--text) !important;
  font-size: 13px !important;
  font-weight: 800 !important;
}

#batch-limit > div,
#batch-limit .wrap,
#batch-limit [data-testid="dropdown"] {
  min-height: 46px !important;
  border-radius: 14px !important;
  background: #ffffff !important;
}

#batch-limit label,
#batch-limit .label-wrap,
#batch-limit span {
  font-size: 12px !important;
  line-height: 1.35 !important;
}

#batch-limit p,
#batch-limit small {
  color: var(--muted) !important;
  font-size: 11px !important;
  line-height: 1.35 !important;
  margin: 2px 0 8px !important;
  max-width: 260px !important;
}

#batch-limit [role="listbox"],
#batch-limit .options,
#batch-limit .dropdown-options,
#batch-limit [class*="options"],
#batch-limit [class*="Options"] {
  background: #ffffff !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  box-shadow: 0 14px 34px rgba(15,23,80,0.12) !important;
  overflow: hidden !important;
}

#batch-limit [role="option"],
#batch-limit .option,
#batch-limit [class*="option"],
#batch-limit [class*="Option"] {
  background: #ffffff !important;
  color: var(--text) !important;
  font-size: 13px !important;
  font-weight: 800 !important;
  min-height: 34px !important;
}

#batch-limit [role="option"]:hover,
#batch-limit .option:hover,
#batch-limit [class*="option"]:hover,
#batch-limit [class*="Option"]:hover {
  background: #edf2ff !important;
  color: var(--brand) !important;
}

/* Gradio mounts dropdown menus outside the component, so keep all popovers light. */
.gradio-container [role="listbox"],
.gradio-container [role="menu"],
body [role="listbox"],
body [role="menu"],
body .options,
body .dropdown-options,
body [class*="options"],
body [class*="Options"] {
  background: #ffffff !important;
  color: var(--text) !important;
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  box-shadow: 0 18px 40px rgba(15,23,80,0.14) !important;
}

.gradio-container [role="option"],
body [role="option"],
body .option,
body [class*="option"],
body [class*="Option"] {
  background: #ffffff !important;
  color: var(--text) !important;
  font-size: 13px !important;
  font-weight: 800 !important;
}

.gradio-container [role="option"]:hover,
body [role="option"]:hover,
body .option:hover,
body [class*="option"]:hover,
body [class*="Option"]:hover {
  background: #edf2ff !important;
  color: var(--brand) !important;
}

#analysis-button {
  height: 46px !important;
  min-height: 46px !important;
  width: 100% !important;
  max-width: 260px !important;
  padding: 0 18px !important;
  border-radius: 999px !important;
  border: 0 !important;
  background: linear-gradient(135deg, #4f46e5, #06b6d4) !important;
  color: #ffffff !important;
  font-size: 13px !important;
  font-weight: 900 !important;
  box-shadow: 0 12px 28px rgba(79,70,229,0.22) !important;
}

#analysis-button:hover {
  opacity: 0.9 !important;
  transform: translateY(-1px) !important;
}

#batch-table {
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  overflow: hidden !important;
  background: #ffffff !important;
  color: #111827 !important;
  box-shadow: 0 10px 28px rgba(15,23,80,0.05) !important;
  min-height: 104px !important;
  max-height: 420px !important;
}

#batch-table *,
#batch-table table,
#batch-table .table-wrap,
#batch-table .wrap,
#batch-table .dataframe,
#batch-table [class*="table"],
#batch-table [class*="dataframe"] {
  font-family: "Plus Jakarta Sans", Inter, sans-serif !important;
  color: #111827 !important;
}

#batch-table .table-wrap,
#batch-table [class*="table-wrap"],
#batch-table [class*="dataframe"] {
  background: #ffffff !important;
}

#batch-table table {
  width: 100% !important;
  table-layout: fixed !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
}

#batch-table th {
  background: var(--surface-muted) !important;
  color: var(--text) !important;
  font-weight: 900 !important;
  font-size: 12px !important;
  line-height: 1.35 !important;
  padding: 11px 12px !important;
  border-color: var(--line) !important;
  white-space: normal !important;
}

#batch-table td {
  background: #ffffff !important;
  color: #111827 !important;
  vertical-align: top !important;
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
  line-height: 1.45 !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  padding: 10px 12px !important;
  border-color: var(--line) !important;
  min-height: 44px !important;
  height: auto !important;
}

#batch-table tbody tr,
#batch-table tbody tr:nth-child(odd),
#batch-table tbody tr:nth-child(even) {
  background: #ffffff !important;
  color: #111827 !important;
  min-height: 44px !important;
}

#batch-table tbody tr:nth-child(even) td {
  background: #f9fbff !important;
}

#batch-table tbody tr:hover td {
  background: #edf2ff !important;
}

#batch-table th:nth-child(1),
#batch-table td:nth-child(1) {
  width: 64px !important;
  text-align: center !important;
}

#batch-table th:nth-child(2),
#batch-table td:nth-child(2) {
  width: 190px !important;
}

#batch-table th:nth-child(5),
#batch-table td:nth-child(5) {
  width: 200px !important;
}

#batch-table th:nth-child(6),
#batch-table td:nth-child(6) {
  width: 160px !important;
  white-space: nowrap !important;
}

#batch-table [role="gridcell"],
#batch-table [role="columnheader"],
#batch-table [class*="cell"],
#batch-table [class*="Cell"] {
  background-color: #ffffff !important;
  color: #111827 !important;
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  line-height: 1.45 !important;
  min-height: 42px !important;
}

#batch-table [role="columnheader"] {
  background-color: var(--surface-muted) !important;
  font-weight: 900 !important;
}

#batch-table [class*="dark"],
#batch-table [style*="rgb(17, 24, 39)"],
#batch-table [style*="#111827"] {
  background: #ffffff !important;
  color: #111827 !important;
}

#batch-table .empty,
#batch-table [class*="empty"],
#batch-table [class*="Empty"] {
  min-height: 52px !important;
  background: #ffffff !important;
  color: var(--muted) !important;
  font-size: 12px !important;
  font-weight: 700 !important;
}

#batch-summary {
  margin-top: 14px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(120px, 1fr));
  gap: 10px;
}

.summary-item {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: #ffffff;
  padding: 12px 13px;
  box-shadow: 0 6px 18px rgba(15,23,80,0.04);
}

.summary-item span {
  display: block;
  color: var(--muted);
  font-size: 11px;
  font-weight: 800;
  margin-bottom: 6px;
}

.summary-item b {
  color: #111827;
  font-size: 16px;
  font-weight: 900;
}

.summary-item.success b { color: #047857; }
.summary-item.danger b { color: #b91c1c; }
.summary-item.warning b { color: #92400e; }

.summary-note {
  margin-top: 10px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--surface-soft);
  color: #111827;
  font-size: 12.5px;
  font-weight: 700;
  line-height: 1.55;
  padding: 12px 14px;
}

/* ── Pipeline footer ── */
.pipeline-footer {
  text-align: center;
  color: var(--muted);
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: 0.05em;
  padding: 6px 0 10px;
}

/* ── Responsive ── */
@media (max-width: 860px) {
  .layout { grid-template-columns: 1fr !important; }
  .analysis-controls { width: 100% !important; }
  .analysis-action-row { grid-template-columns: minmax(0, 1fr) !important; }
  #analysis-button { width: 100% !important; max-width: none !important; }
  .summary-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
  .hero-inner { grid-template-columns: 1fr; }
  .hero-visual { display: none; }
  .hero { padding: 32px 24px; }
  .card { padding: 20px !important; }
}
@media (max-width: 540px) {
  .gradio-container { padding: 0 10px 36px !important; }
  .hero { padding: 24px 18px; }
  .hero-title { font-size: 24px; }
  .hero-badges { display: none; }
  .summary-grid { grid-template-columns: 1fr; }
}
"""


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(theme=theme, css=css, title="S2S Voice Chatbot") as demo:
    with gr.Column(elem_classes="app-shell"):

        # ── Hero ─────────────────────────────────────────────────────────────
        gr.HTML("""
        <section class="hero">
            <div class="hero-inner">
                <div>
                    <div class="hero-eyebrow">AI Voice Assistant</div>
                    <h1 class="hero-title">
                        <span class="word-sub">Multilingual Code-Switching</span>
                        <span class="word-main">Speech-to-Speech</span>
                    </h1>
                    <p class="hero-desc">
                        Voice chatbot interaktif &mdash; rekam suara, analisis code-switching,
                        respons LLM, dan sintesis suara dalam satu pipeline terintegrasi.
                    </p>
                    <div class="hero-badges">
                        <span class="hero-badge">
                            <span class="hero-badge-dot" style="background:#4f46e5"></span>
                            Speech-to-Text
                        </span>
                        <span class="hero-badge">
                            <span class="hero-badge-dot" style="background:#06b6d4"></span>
                            Normalisasi
                        </span>
                        <span class="hero-badge">
                            <span class="hero-badge-dot" style="background:#8b5cf6"></span>
                            Language Detection
                        </span>
                        <span class="hero-badge">
                            <span class="hero-badge-dot" style="background:#10b981"></span>
                            LLM Response
                        </span>
                        <span class="hero-badge">
                            <span class="hero-badge-dot" style="background:#f59e0b"></span>
                            TTS Synthesis
                        </span>
                    </div>
                </div>
                <div class="hero-visual" aria-hidden="true">
                    <div class="hero-dots">
                        <span></span><span></span><span></span>
                        <span></span><span></span><span></span>
                        <span></span><span></span><span></span>
                    </div>
                    <div class="hero-visual-ring"></div>
                    <div class="hero-visual-ring-2"></div>
                    <div class="hero-visual-core">
                        <svg viewBox="0 0 24 24">
                            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 1 0 6 0V5a3 3 0 0 0-3-3Z"/>
                            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                            <path d="M12 19v3"/>
                            <path d="M8 22h8"/>
                        </svg>
                    </div>
                </div>
            </div>
        </section>
        """)

        # ── Two-column layout ─────────────────────────────────────────────────
        with gr.Row(elem_classes="layout"):

            # ── LEFT: Input Panel ─────────────────────────────────────────────
            with gr.Column(elem_classes="card"):
                gr.HTML("""
                <div class="panel-head">
                    <div>
                        <h2 class="panel-title">Input Rekaman Suara</h2>
                        <p class="panel-copy">Rekam ujaran, pilih mode respons, lalu jalankan pipeline.</p>
                    </div>
                    <span class="chip">Ready</span>
                </div>
                """)

                gr.HTML("""
                <div class="voice-card">
                    <div class="voice-visual" aria-hidden="true">
                        <div class="mic-disc">
                            <svg viewBox="0 0 24 24">
                                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 1 0 6 0V5a3 3 0 0 0-3-3Z"/>
                                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                                <path d="M12 19v3"/>
                                <path d="M8 22h8"/>
                            </svg>
                        </div>
                    </div>
                </div>
                """)

                gr.HTML('<div class="audio-field-label">Rekam / unggah ujaran Anda</div>')
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Rekam / unggah ujaran Anda",
                    show_label=False,
                    elem_id="audio-input",
                )

                with gr.Column(elem_classes="mode-panel"):
                    mode_select = gr.Radio(
                        choices=["preserve", "normalized"],
                        value="preserve",
                        label="Mode Respons Sistem",
                        info="preserve = bahasa campuran  |  normalized = Bahasa Indonesia formal",
                        elem_id="mode-select",
                    )

                gr.HTML('<div class="send-zone">')
                submit_btn = gr.Button("Proses Pipeline", variant="primary", elem_id="submit-button")
                gr.HTML('<span class="send-hint">Selesai merekam atau memilih file? Klik tombol di atas.</span></div>')

                gr.HTML("""
                <div class="help-card">
                    Cara pakai: rekam langsung, atau unggah file audio untuk diuji.
                    Setelah itu klik Proses Pipeline dan hasil akan muncul di dashboard kanan.
                </div>
                """)

            # ── RIGHT: Pipeline Dashboard ─────────────────────────────────────
            with gr.Column(elem_classes="card"):
                gr.HTML("""
                <div class="panel-head">
                    <div>
                        <h2 class="panel-title">Dashboard Visualisasi Pipeline</h2>
                        <p class="panel-copy">Setiap tahap terisi setelah backend selesai memproses audio.</p>
                    </div>
                    <span class="chip">Live</span>
                </div>
                """)

                # Step 1
                with gr.Column(elem_classes="flow-card"):
                    gr.HTML("""
                    <div class="step-head">
                        <span class="step-number">1</span>
                        <h3 class="flow-title">Speech-to-Text Transcription</h3>
                    </div>
                    """)
                    out_user_text = gr.Textbox(
                        label="Hasil Transkripsi Suara",
                        interactive=False,
                        placeholder="Menunggu transkripsi...",
                    )

                # Step 2
                with gr.Column(elem_classes="flow-card"):
                    gr.HTML("""
                    <div class="step-head">
                        <span class="step-number">2</span>
                        <h3 class="flow-title">Normalisasi Kata Kolokial</h3>
                    </div>
                    """)
                    out_normalized_text = gr.Textbox(
                        label="Hasil Normalisasi Leksikon",
                        interactive=False,
                        placeholder="Menunggu normalisasi...",
                    )

                # Step 3
                with gr.Column(elem_classes="flow-card"):
                    gr.HTML("""
                    <div class="step-head">
                        <span class="step-number">3</span>
                        <h3 class="flow-title">Deteksi Bahasa dan Code-Switching</h3>
                    </div>
                    """)
                    out_language_tags = gr.HTML(
                        value=language_tags_markup(None),
                        label="Tagging Kata Multibahasa",
                    )
                    out_ratios = gr.Textbox(
                        label="Proporsi Bahasa dalam Ujaran",
                        interactive=False,
                        placeholder="Menunggu analisis...",
                    )

                # Step 4
                with gr.Column(elem_classes="flow-card"):
                    gr.HTML("""
                    <div class="step-head">
                        <span class="step-number">4</span>
                        <h3 class="flow-title">Kontekstual Respons LLM</h3>
                    </div>
                    """)
                    out_llm_response = gr.Textbox(
                        label="Respons Teks",
                        interactive=False,
                        placeholder="Menunggu respons LLM...",
                        lines=4,
                    )

                # Step 5
                with gr.Column(elem_classes="flow-card"):
                    gr.HTML("""
                    <div class="step-head">
                        <span class="step-number">5</span>
                        <h3 class="flow-title">Sintesis Suara TTS</h3>
                    </div>
                    <p class="flow-note">
                        Setelah pipeline sukses, audio balasan muncul di player ini.
                        Klik play untuk mendengarkan suara asisten.
                    </p>
                    """)
                    audio_output = gr.Audio(
                        type="filepath",
                        label="Balasan Suara Asisten",
                        interactive=False,
                        elem_id="audio-output",
                    )

                # Status
                status_html = gr.HTML(status_markup())
                status_detail = gr.Textbox(
                    label="Log Status Pipeline",
                    interactive=False,
                    value="Idle",
                    elem_id="status-detail",
                )

        # ── Batch Pipeline Analysis ─────────────────────────────────────────
        with gr.Column(elem_classes="card analysis-card"):
            gr.HTML("""
            <div class="panel-head">
                <div>
                    <h2 class="panel-title">Analisis Pipeline</h2>
                    <p class="panel-copy">
                        Uji output pipeline untuk audio di folder data/audio dengan batas data terpilih.
                    </p>
                </div>
                <span class="chip">Batch</span>
            </div>
            """)

            with gr.Column(elem_classes="analysis-controls"):
                gr.HTML("""
                <div class="analysis-control-copy">
                    <h3>Jumlah Data</h3>
                    <p>
                        Pilih jumlah audio yang ingin dianalisis. Maksimal 500 data.
                        Gunakan pilihan ini untuk membatasi proses batch agar tetap ringan dan mudah dipantau.
                    </p>
                </div>
                """)
                with gr.Row(elem_classes="analysis-action-row"):
                    batch_limit = gr.Dropdown(
                        choices=BATCH_LIMIT_CHOICES,
                        value=None,
                        label="Pilih Jumlah Data",
                        show_label=False,
                        elem_id="batch-limit",
                    )
                    batch_btn = gr.Button(
                        "Analisis Pipeline",
                        variant="primary",
                        elem_id="analysis-button",
                    )

            batch_status = gr.HTML(
                status_markup("idle", "Pilih jumlah data, lalu jalankan analisis pipeline.")
            )
            batch_table = gr.Dataframe(
                headers=BATCH_TABLE_HEADERS,
                value=pd.DataFrame(columns=BATCH_TABLE_HEADERS),
                datatype=["number", "str", "str", "str", "str", "str"],
                type="pandas",
                row_count=(0, "dynamic"),
                col_count=(6, "fixed"),
                max_height=520,
                wrap=True,
                line_breaks=True,
                column_widths=["64px", "190px", "28%", "32%", "200px", "160px"],
                show_search="none",
                show_row_numbers=False,
                max_chars=300,
                interactive=False,
                label="Hasil Analisis Pipeline",
                elem_id="batch-table",
            )
            batch_summary = gr.HTML(
                label="Ringkasan Analisis",
                value=summary_markup(),
                elem_id="batch-summary",
            )

        # ── Footer ────────────────────────────────────────────────────────────
        gr.HTML("<div class='pipeline-footer'>STT  —  Normalisasi  —  Language Tagging  —  LLM  —  TTS</div>")

    # ── Event wiring ─────────────────────────────────────────────────────────
    submit_btn.click(
        fn=voice_chat_pipeline,
        inputs=[audio_input, mode_select],
        outputs=[
            audio_output,
            out_user_text,
            out_normalized_text,
            out_language_tags,
            out_ratios,
            out_llm_response,
            status_html,
            status_detail,
        ],
    )

    batch_btn.click(
        fn=run_batch_pipeline_analysis,
        inputs=[batch_limit, mode_select],
        outputs=[batch_table, batch_status, batch_summary],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
