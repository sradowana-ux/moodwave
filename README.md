# Multimodal Emotion Recognition for Mood-Based Music Recommendation

A research pipeline that classifies emotional mood from **text** and **speech** and combines both signals to support mood-based music recommendations. Built as a university research project at the University of Hull.

---

## What It Does

The system predicts one of five mood categories — **anger, fear, joy, neutral, sadness** — using three experimental settings:

| Setting | Modality | Model |
|---------|----------|-------|
| Text baseline | Reddit comments (GoEmotions) | `SamLowe/roberta-base-go_emotions` (RoBERTa) |
| Audio baseline | Acted speech (RAVDESS) | `facebook/wav2vec2-base` fine-tuned |
| Late fusion | Text + audio combined | Weighted probability averaging |

---

## Results

All metrics are weighted averages over 5 mood classes. The audio model was evaluated on a held-out actor split (actors 1, 12, 15, 24); the text model on a stratified sample of the GoEmotions test set (n=500); the multimodal setting on a proxy evaluation set (n=250) that pairs text and audio samples by matching mood label — see Limitations.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Audio baseline (Wav2Vec2-RAVDESS) | 0.841 | 0.873 | 0.841 | 0.844 |
| Multimodal — best fixed fusion (text=0.2, audio=0.8) | **0.852** | **0.870** | **0.852** | **0.850** |
| Multimodal — confidence-weighted fusion | 0.848 | 0.870 | 0.848 | 0.843 |
| Text baseline (RoBERTa-GoEmotions) | 0.650 | 0.719 | 0.650 | 0.658 |

**Audio 5-fold cross-validation (actor-grouped):** F1 = 0.69 / 0.77 / 0.86 / 0.73 / 0.76 — mean ≈ 0.76, std ≈ 0.06.

The audio model substantially outperforms the text model (F1 0.84 vs 0.66). Late fusion provides a small additional gain (+0.006 F1), and the optimal fusion weights (text=0.2, audio=0.8) confirm that the audio signal dominates. The text model struggles most with **fear** (50% recall), which it frequently misclassifies as neutral.

---

## Tech Stack

- **Python** 3.x, PyTorch, HuggingFace Transformers / Datasets
- **Text:** `SamLowe/roberta-base-go_emotions` — zero-shot inference, no fine-tuning
- **Audio:** `facebook/wav2vec2-base` — fine-tuned on RAVDESS with 5-epoch training
- **Audio features:** librosa (resampling, waveform loading), scikit-learn (evaluation)
- **Training:** HuggingFace `Trainer` with early stopping on validation F1
- **Evaluation:** stratified sampling, actor-grouped CV, confusion matrices

---

## Datasets

- **[GoEmotions](https://huggingface.co/datasets/go_emotions)** (Demszky et al., 2020) — 58k Reddit comments, 27 emotion labels, mapped to 5-class mood space
- **[RAVDESS](https://zenodo.org/record/1188976)** — 2,496 acted speech recordings from 24 professional actors, 8 emotion classes mapped to 5-class mood space

---

## Limitations

This project is a research prototype with several constraints worth understanding:

1. **RAVDESS is acted speech.** The audio model was trained and tested on professional actors performing emotions in a studio setting. Real-world speech emotion recognition from natural, spontaneous audio would likely yield lower performance.

2. **Proxy multimodal evaluation.** The fusion system is evaluated on cross-dataset pairs — text from GoEmotions and audio from RAVDESS — matched only by mood label, not by utterance. This is not a truly paired multimodal evaluation; it is a design choice made necessary by the absence of a jointly annotated text+speech dataset in the same five-class mood space.

3. **Best fusion weight is in-sample.** The optimal fixed fusion weight (text=0.2, audio=0.8) was selected on the same proxy evaluation set used for reporting. This makes the best-fixed-weight result slightly optimistic; the confidence-weighted fusion result is a cleaner comparison.

4. **High variance in audio cross-validation.** Fold F1 ranges from 0.69 to 0.86. Actor identity is a strong source of variation, which is expected with a small 24-actor dataset.

5. **Text model is zero-shot.** The RoBERTa model is used off-the-shelf without fine-tuning, which limits its performance on the five-class mapping — especially for mood categories like fear that map awkwardly from GoEmotions labels.

---

## Project Structure

```
MER_Project/
├── audio_data/             # Extracted RAVDESS audio files
├── audio_final_model/      # Saved Wav2Vec2 model weights
│   └── best_model/
└── report_figures/         # Evaluation plots (confusion matrices, F1 comparisons)

Research_project(Radowana)hull26.ipynb   # Main notebook
```

---

## How to Run

1. Install dependencies (pinned versions in Cell 1 of the notebook)
2. Download the [RAVDESS dataset](https://zenodo.org/record/1188976) and place `archive.zip` in `MER_Project/`
3. Run all cells in order — GoEmotions loads automatically via HuggingFace Datasets
4. GPU strongly recommended for audio fine-tuning (tested on CUDA)

---

## References

- Demszky, D., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *ACL 2020*.
- Baevski, A., et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *NeurIPS 2020*.
- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLOS ONE*.
