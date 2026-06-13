"""
==============================================================================
  TEXT MODEL FINE-TUNING  —  add these cells to the notebook after Section 9
==============================================================================

Currently the text model uses SamLowe/roberta-base-go_emotions zero-shot
with probability aggregation across mapped label groups. This cell fine-tunes
a RoBERTa model directly on the 5-class mood mapping, which should improve
F1 from ~0.66 to a higher value (typically 0.75–0.82 on this setup).

Estimated runtime: ~15–25 min on GPU (3 epochs over 43k training samples).
==============================================================================

CELL MARKDOWN:
## 9.2 (Optional) Fine-tune RoBERTa on the 5-class mood mapping

The zero-shot approach in Section 9.1 achieves F1 = 0.658 on the held-out
GoEmotions test split. The root cause is that the probability aggregation
over 28 → 5 class mappings introduces noise: some 28-class labels map
ambiguously (e.g. `annoyance` and `anger` both map to anger, but so does
`disgust` in some schemes). Fine-tuning RoBERTa directly on five-class
supervision removes this aggregation step and lets the model learn decision
boundaries specific to the target mood space.

This cell builds a training set from GoEmotions by applying the same
`MOOD_MAP` from Section 8, then fine-tunes `roberta-base` for 3 epochs.
Early stopping on validation F1 prevents overfitting.
"""

# ── 9.2  Fine-tune RoBERTa on 5-class mood labels ─────────────────────────
import os, gc, random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ── Configuration ─────────────────────────────────────────────────────────
FINETUNE_MODEL_NAME = "roberta-base"
FINETUNE_OUTPUT_DIR = PROJECT_DIR / "text_finetuned_model"
FINETUNE_BEST_DIR   = FINETUNE_OUTPUT_DIR / "best"
FINETUNE_EPOCHS     = 3
FINETUNE_LR         = 2e-5
FINETUNE_BATCH      = 32
MAX_TEXT_LEN        = 128

FINETUNE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Build 5-class training / validation DataFrames from GoEmotions ────────
def goemotions_to_mood_df(dataset_split, max_rows=None):
    """Flatten GoEmotions to single-label 5-class mood rows, dropping ambiguous."""
    rows = []
    for i in range(len(dataset_split) if max_rows is None else min(len(dataset_split), max_rows)):
        item = dataset_split[i]
        mood = map_goemotions_multilabel_to_single_mood(item["labels"])
        if mood is not None:
            rows.append({"text": item["text"], "label": label2id[mood]})
    return pd.DataFrame(rows)

print("Building fine-tune splits from GoEmotions…")
train_ft_df = goemotions_to_mood_df(go["train"])
val_ft_df   = goemotions_to_mood_df(go["validation"])
test_ft_df  = goemotions_to_mood_df(go["test"])

print(f"  train : {len(train_ft_df):,}  |  val : {len(val_ft_df):,}  |  test : {len(test_ft_df):,}")
print("  Mood distribution (train):")
print(train_ft_df["label"].map(id2label).value_counts().to_string())

# ── Tokenise ───────────────────────────────────────────────────────────────
ft_tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL_NAME)

def tokenise(batch):
    return ft_tokenizer(batch["text"], truncation=True, max_length=MAX_TEXT_LEN)

hf_train = Dataset.from_pandas(train_ft_df)
hf_val   = Dataset.from_pandas(val_ft_df)
hf_test  = Dataset.from_pandas(test_ft_df)

hf_train = hf_train.map(tokenise, batched=True, remove_columns=["text"])
hf_val   = hf_val.map(tokenise,   batched=True, remove_columns=["text"])
hf_test  = hf_test.map(tokenise,  batched=True, remove_columns=["text"])

collator = DataCollatorWithPadding(tokenizer=ft_tokenizer)

# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="weighted", zero_division=0),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall":    recall_score(labels, preds, average="weighted", zero_division=0),
    }

# ── Model ──────────────────────────────────────────────────────────────────
ft_model = AutoModelForSequenceClassification.from_pretrained(
    FINETUNE_MODEL_NAME,
    num_labels=len(TARGET_LABELS),
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir=str(FINETUNE_OUTPUT_DIR),
    num_train_epochs=FINETUNE_EPOCHS,
    per_device_train_batch_size=FINETUNE_BATCH,
    per_device_eval_batch_size=FINETUNE_BATCH,
    learning_rate=FINETUNE_LR,
    weight_decay=0.01,
    warmup_ratio=0.06,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=200,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer_ft = Trainer(
    model=ft_model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    tokenizer=ft_tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\nFine-tuning RoBERTa on 5-class mood labels…")
trainer_ft.train()
trainer_ft.save_model(str(FINETUNE_BEST_DIR))
print(f"Best model saved to: {FINETUNE_BEST_DIR}")

# ── Evaluate on held-out test split ───────────────────────────────────────
print("\n── Fine-tuned text model TEST results ────────────────────────────────")
ft_test_results = trainer_ft.evaluate(hf_test)
print(pd.Series(ft_test_results).round(4).to_string())

# Compare against zero-shot baseline
improvement_f1 = ft_test_results.get("eval_f1", 0) - text_results["f1"]
print(f"\nF1 improvement over zero-shot baseline: {improvement_f1:+.4f}")

# ── Replace predict_text_mood with fine-tuned version ─────────────────────
from transformers import pipeline as hf_pipeline

ft_pipeline = hf_pipeline(
    "text-classification",
    model=str(FINETUNE_BEST_DIR),
    tokenizer=str(FINETUNE_BEST_DIR),
    device=0 if DEVICE == "cuda" else -1,
    return_all_scores=True,
    truncation=True,
    max_length=MAX_TEXT_LEN,
)

def predict_text_mood_finetuned(text):
    """Predict mood using the fine-tuned 5-class model (no label aggregation needed)."""
    text = clean_text(text)
    out  = ft_pipeline(text)[0]  # list of {label, score}
    scores = {item["label"]: item["score"] for item in out}
    best   = max(scores, key=scores.get)
    return {
        "mood":       best,
        "confidence": scores[best],
        "certainty":  scores[best] - sorted(scores.values())[-2],
        "scores":     scores,
    }

# Quick sanity check
sample_text = "I finally got the job offer, I can't believe it!"
print(f"\nSample: '{sample_text}'")
print(predict_text_mood_finetuned(sample_text))
