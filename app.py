"""
Multimodal Emotion Recognition — Interactive Demo
==================================================
Run with:  python app.py
Then open: http://localhost:7860

Requirements:
    pip install gradio transformers torch librosa soundfile

Model paths expected (relative to this file):
    MER_Project/audio_final_model/best_model/   <- fine-tuned Wav2Vec2
    MER_Project/text_finetuned_model/best/      <- fine-tuned RoBERTa  (optional)

If the fine-tuned text model doesn't exist the app falls back to the
zero-shot SamLowe/roberta-base-go_emotions model automatically.
"""

import os
import tempfile
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch

# ── Suppress noisy warnings ───────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

from transformers import (
    AutoFeatureExtractor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Wav2Vec2ForSequenceClassification,
    pipeline as hf_pipeline,
)

# ── Config ────────────────────────────────────────────────────────────────
TARGET_LABELS   = ["anger", "fear", "joy", "neutral", "sadness"]
label2id        = {l: i for i, l in enumerate(TARGET_LABELS)}
id2label        = {i: l for l, i in label2id.items()}
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_DIR             = Path("MER_Project")
AUDIO_MODEL_DIR         = PROJECT_DIR / "audio_final_model" / "best_model"
TEXT_FINETUNED_DIR      = PROJECT_DIR / "text_finetuned_model" / "best"
TEXT_ZERO_SHOT_NAME     = "SamLowe/roberta-base-go_emotions"
TARGET_SR               = 16_000
MAX_SAMPLES             = int(TARGET_SR * 6.0)

# GoEmotions 28-class → 5-class mood mapping (same as notebook)
MOOD_MAP = {
    "admiration": "joy",      "amusement": "joy",
    "anger": "anger",         "annoyance": "anger",
    "approval": "joy",        "caring": "joy",
    "confusion": "neutral",   "curiosity": "neutral",
    "desire": "joy",          "disappointment": "sadness",
    "disapproval": "anger",   "disgust": "anger",
    "embarrassment": "sadness","excitement": "joy",
    "fear": "fear",           "gratitude": "joy",
    "grief": "sadness",       "joy": "joy",
    "love": "joy",            "nervousness": "fear",
    "optimism": "joy",        "pride": "joy",
    "realization": "neutral", "relief": "joy",
    "remorse": "sadness",     "sadness": "sadness",
    "surprise": "neutral",    "neutral": "neutral",
}

MOOD_EMOJI = {
    "anger":   "😠",
    "fear":    "😨",
    "joy":     "😊",
    "neutral": "😐",
    "sadness": "😢",
}

# ── Load models ───────────────────────────────────────────────────────────
print("Loading models…")

# Audio
if AUDIO_MODEL_DIR.exists():
    _audio_feature_extractor = AutoFeatureExtractor.from_pretrained(str(AUDIO_MODEL_DIR))
    _audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(str(AUDIO_MODEL_DIR))
    _audio_model.eval().to(DEVICE)
    AUDIO_READY = True
    print(f"  Audio model loaded from {AUDIO_MODEL_DIR}")
else:
    AUDIO_READY = False
    print(f"  WARNING: Audio model not found at {AUDIO_MODEL_DIR}. Audio tab disabled.")

# Text
if TEXT_FINETUNED_DIR.exists():
    _text_pipe = hf_pipeline(
        "text-classification",
        model=str(TEXT_FINETUNED_DIR),
        tokenizer=str(TEXT_FINETUNED_DIR),
        device=0 if DEVICE == "cuda" else -1,
        return_all_scores=True,
        truncation=True,
        max_length=128,
    )
    TEXT_MODE = "finetuned"
    print(f"  Text model loaded (fine-tuned) from {TEXT_FINETUNED_DIR}")
else:
    _text_pipe = hf_pipeline(
        "text-classification",
        model=TEXT_ZERO_SHOT_NAME,
        device=0 if DEVICE == "cuda" else -1,
        return_all_scores=True,
        truncation=True,
        max_length=128,
    )
    TEXT_MODE = "zero-shot"
    print(f"  Text model loaded (zero-shot): {TEXT_ZERO_SHOT_NAME}")


# ── Prediction helpers ────────────────────────────────────────────────────
def predict_text(text: str) -> dict:
    """Return mood prediction dict from text input."""
    text = text.strip()
    if not text:
        return None
    raw = _text_pipe(text)[0]
    if TEXT_MODE == "finetuned":
        scores = {item["label"]: item["score"] for item in raw}
    else:
        # Aggregate 28 GoEmotions labels into 5 mood scores
        mood_scores = {m: 0.0 for m in TARGET_LABELS}
        for item in raw:
            mood = MOOD_MAP.get(item["label"])
            if mood:
                mood_scores[mood] += item["score"]
        total = sum(mood_scores.values()) or 1.0
        scores = {m: v / total for m, v in mood_scores.items()}
    best = max(scores, key=scores.get)
    return {"mood": best, "confidence": scores[best], "scores": scores}


def predict_audio(audio_path: str) -> dict:
    """Return mood prediction dict from audio file path."""
    if not AUDIO_READY or audio_path is None:
        return None
    waveform, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    waveform = waveform[:MAX_SAMPLES]
    inputs = _audio_feature_extractor(
        waveform, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
    ).to(DEVICE)
    with torch.no_grad():
        logits = _audio_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    scores = {id2label[i]: p for i, p in enumerate(probs)}
    best = max(scores, key=scores.get)
    return {"mood": best, "confidence": scores[best], "scores": scores}


def fuse(text_pred, audio_pred, text_weight=0.2, audio_weight=0.8):
    """Confidence-weighted or fixed-weight late fusion."""
    if text_pred is None and audio_pred is None:
        return None
    if text_pred is None:
        return audio_pred
    if audio_pred is None:
        return text_pred
    fused_scores = {}
    for mood in TARGET_LABELS:
        fused_scores[mood] = (text_weight  * text_pred["scores"].get(mood, 0) +
                              audio_weight * audio_pred["scores"].get(mood, 0))
    best = max(fused_scores, key=fused_scores.get)
    return {"mood": best, "confidence": fused_scores[best], "scores": fused_scores}


def scores_to_bar_data(scores: dict):
    """Return (labels, values) lists sorted by score descending."""
    items = sorted(scores.items(), key=lambda x: -x[1])
    labels = [f"{MOOD_EMOJI.get(m, '')} {m}" for m, _ in items]
    values = [round(v * 100, 1) for _, v in items]
    return labels, values


# ── Gradio interface functions ────────────────────────────────────────────
def run_text_only(text):
    pred = predict_text(text)
    if pred is None:
        return "Please enter some text.", {}
    labels, values = scores_to_bar_data(pred["scores"])
    mood = pred["mood"]
    emoji = MOOD_EMOJI.get(mood, "")
    result = f"{emoji} **{mood.upper()}** ({pred['confidence']*100:.1f}% confidence)"
    return result, gr.BarPlot.update(
        value={"mood": labels, "score": values},
        x="mood", y="score",
    )


def run_audio_only(audio):
    if not AUDIO_READY:
        return "⚠️ Audio model not loaded. Check that MER_Project/audio_final_model/best_model/ exists.", {}
    pred = predict_audio(audio)
    if pred is None:
        return "Please upload an audio file.", {}
    labels, values = scores_to_bar_data(pred["scores"])
    mood = pred["mood"]
    emoji = MOOD_EMOJI.get(mood, "")
    result = f"{emoji} **{mood.upper()}** ({pred['confidence']*100:.1f}% confidence)"
    return result, gr.BarPlot.update(
        value={"mood": labels, "score": values},
        x="mood", y="score",
    )


def run_multimodal(text, audio, text_w, audio_w):
    text_pred  = predict_text(text) if text and text.strip() else None
    audio_pred = predict_audio(audio) if (AUDIO_READY and audio) else None

    if text_pred is None and audio_pred is None:
        return "Please provide text, audio, or both.", {}, "—", "—"

    fused = fuse(text_pred, audio_pred, text_weight=text_w, audio_weight=audio_w)
    labels, values = scores_to_bar_data(fused["scores"])
    mood  = fused["mood"]
    emoji = MOOD_EMOJI.get(mood, "")
    result = f"{emoji} **{mood.upper()}** ({fused['confidence']*100:.1f}% confidence)"

    text_summary  = (f"{MOOD_EMOJI.get(text_pred['mood'],'')} {text_pred['mood']} "
                     f"({text_pred['confidence']*100:.0f}%)" if text_pred else "—")
    audio_summary = (f"{MOOD_EMOJI.get(audio_pred['mood'],'')} {audio_pred['mood']} "
                     f"({audio_pred['confidence']*100:.0f}%)" if audio_pred else "—")

    return result, gr.BarPlot.update(
        value={"mood": labels, "score": values},
        x="mood", y="score",
    ), text_summary, audio_summary


# ── Build UI ──────────────────────────────────────────────────────────────
CSS = """
h1 { font-size: 1.6rem; font-weight: 700; }
.result-box { font-size: 1.3rem; padding: 12px; border-radius: 8px;
              background: var(--color-background-secondary); }
"""

with gr.Blocks(title="Mood Detector", css=CSS) as demo:
    gr.Markdown(
        "# 🎵 Multimodal Emotion Recognition\n"
        "Predict mood from **text**, **speech**, or **both combined** using "
        "RoBERTa (text) + Wav2Vec2 (audio) with late fusion.\n\n"
        f"> Text model: **{TEXT_MODE}** &nbsp;|&nbsp; "
        f"Audio model: **{'loaded ✅' if AUDIO_READY else 'not found ⚠️'}**"
    )

    with gr.Tabs():
        # ── Tab 1: Text only ─────────────────────────────────────────────
        with gr.Tab("📝 Text"):
            with gr.Row():
                txt_input = gr.Textbox(
                    label="Enter text",
                    placeholder="How are you feeling? Type anything…",
                    lines=4,
                )
            txt_btn    = gr.Button("Predict mood", variant="primary")
            txt_result = gr.Markdown(elem_classes="result-box")
            txt_chart  = gr.BarPlot(
                x="mood", y="score",
                title="Mood probability (%)",
                y_lim=[0, 100],
                width=500,
            )
            txt_btn.click(run_text_only, inputs=txt_input, outputs=[txt_result, txt_chart])
            gr.Examples(
                examples=[
                    ["I finally got the internship offer, I'm over the moon!"],
                    ["I don't really care either way, it is what it is."],
                    ["This keeps happening and I'm so done with it."],
                    ["I keep thinking something terrible is about to happen."],
                    ["I miss how things used to be."],
                ],
                inputs=txt_input,
            )

        # ── Tab 2: Audio only ────────────────────────────────────────────
        with gr.Tab("🎙️ Audio"):
            aud_input  = gr.Audio(label="Upload a WAV/MP3 (up to 6 seconds used)",
                                  type="filepath")
            aud_btn    = gr.Button("Predict mood", variant="primary")
            aud_result = gr.Markdown(elem_classes="result-box")
            aud_chart  = gr.BarPlot(
                x="mood", y="score",
                title="Mood probability (%)",
                y_lim=[0, 100],
                width=500,
            )
            aud_btn.click(run_audio_only, inputs=aud_input, outputs=[aud_result, aud_chart])
            gr.Markdown(
                "> **Note:** The audio model was trained on RAVDESS acted speech. "
                "Performance on natural, spontaneous speech may be lower."
            )

        # ── Tab 3: Multimodal fusion ─────────────────────────────────────
        with gr.Tab("🔀 Multimodal Fusion"):
            with gr.Row():
                mm_text  = gr.Textbox(label="Text input", lines=3,
                                      placeholder="Type something…")
                mm_audio = gr.Audio(label="Audio input", type="filepath")
            with gr.Row():
                text_w  = gr.Slider(0, 1, value=0.2, step=0.05, label="Text weight")
                audio_w = gr.Slider(0, 1, value=0.8, step=0.05, label="Audio weight")
            mm_btn    = gr.Button("Fuse and predict", variant="primary")
            mm_result = gr.Markdown(elem_classes="result-box")
            mm_chart  = gr.BarPlot(
                x="mood", y="score",
                title="Fused mood probability (%)",
                y_lim=[0, 100],
                width=500,
            )
            with gr.Row():
                txt_contrib = gr.Textbox(label="Text-only prediction", interactive=False)
                aud_contrib = gr.Textbox(label="Audio-only prediction", interactive=False)
            mm_btn.click(
                run_multimodal,
                inputs=[mm_text, mm_audio, text_w, audio_w],
                outputs=[mm_result, mm_chart, txt_contrib, aud_contrib],
            )
            gr.Markdown(
                "> Default fusion weights (text=0.2, audio=0.8) are from the research paper. "
                "Adjust the sliders to explore how each modality contributes."
            )

    gr.Markdown(
        "---\n"
        "**Research project:** Multimodal Emotion Recognition for Mood-Based Music Recommendation  \n"
        "University of Hull · 2026"
    )

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
