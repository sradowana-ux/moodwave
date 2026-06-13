"""
==============================================================================
  DEEPER ANALYSIS  —  add these cells to the notebook after Cell 35
==============================================================================

CELL A  (Markdown)
──────────────────
## 17. Deeper Analysis

### 17.1 Per-class error breakdown

The headline metrics hide meaningful differences across mood categories.
This section unpacks per-class recall, precision, and F1 from the confusion
matrices and discusses what drives the most common misclassifications.

──────────────────
CELL B  (Code)
"""

# ── 17.1  Per-class metrics from confusion matrices ────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def per_class_metrics(cm_df):
    """
    Compute per-class precision, recall, and F1 from a labelled confusion matrix.
    cm_df rows  = true labels  (e.g. true_anger)
    cm_df cols  = pred labels  (e.g. pred_anger)
    """
    classes = [c.replace("pred_", "") for c in cm_df.columns]
    rows = []
    for cls in classes:
        tp = cm_df.loc[f"true_{cls}", f"pred_{cls}"]
        fp = cm_df[f"pred_{cls}"].sum() - tp
        fn = cm_df.loc[f"true_{cls}"].sum() - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        rows.append({"class": cls, "support": int(tp + fn),
                     "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)})
    return pd.DataFrame(rows).set_index("class")

text_per_class  = per_class_metrics(text_results["confusion_matrix"])
audio_per_class = per_class_metrics(audio_results["confusion_matrix"])

print("── Text model  (RoBERTa)  per-class ──────────────────────────────────")
print(text_per_class.to_string())
print("\n── Audio model (Wav2Vec2) per-class ──────────────────────────────────")
print(audio_per_class.to_string())

# ── Side-by-side F1 bar chart ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(text_per_class))
w = 0.35
ax.bar(x - w/2, text_per_class["f1"],  w, label="Text (RoBERTa)",  color="#4C72B0", alpha=0.85)
ax.bar(x + w/2, audio_per_class["f1"], w, label="Audio (Wav2Vec2)", color="#DD8452", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(text_per_class.index, fontsize=11)
ax.set_ylabel("F1")
ax.set_title("Per-class F1: Text vs Audio Baseline")
ax.legend()
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
plt.savefig(PROJECT_DIR / "report_figures" / "per_class_f1_comparison.png", dpi=150)
plt.show()
print("Saved: per_class_f1_comparison.png")


"""
──────────────────
CELL C  (Code)
"""
# ── 17.2  Most common misclassifications ──────────────────────────────────
print("── Text model top misclassifications ────────────────────────────────")
cm = text_results["confusion_matrix"]
errors = []
for true_cls in cm.index:
    for pred_cls in cm.columns:
        if true_cls.replace("true_", "") != pred_cls.replace("pred_", ""):
            count = cm.loc[true_cls, pred_cls]
            if count > 0:
                errors.append({
                    "true":  true_cls.replace("true_", ""),
                    "pred":  pred_cls.replace("pred_", ""),
                    "count": int(count),
                    "pct_of_true": round(count / cm.loc[true_cls].sum() * 100, 1),
                })

error_df = (pd.DataFrame(errors)
              .sort_values("count", ascending=False)
              .reset_index(drop=True))
print(error_df.head(10).to_string(index=False))

print("\n── Audio model top misclassifications ───────────────────────────────")
cm_a = audio_results["confusion_matrix"]
errors_a = []
for true_cls in cm_a.index:
    for pred_cls in cm_a.columns:
        if true_cls.replace("true_", "") != pred_cls.replace("pred_", ""):
            count = cm_a.loc[true_cls, pred_cls]
            if count > 0:
                errors_a.append({
                    "true":  true_cls.replace("true_", ""),
                    "pred":  pred_cls.replace("pred_", ""),
                    "count": int(count),
                    "pct_of_true": round(count / cm_a.loc[true_cls].sum() * 100, 1),
                })

error_df_a = (pd.DataFrame(errors_a)
               .sort_values("count", ascending=False)
               .reset_index(drop=True))
print(error_df_a.head(10).to_string(index=False))


"""
──────────────────
CELL D  (Markdown)
──────────────────
## 17.3 Discussion of error patterns

**Text model — fear → neutral (33 % of fear examples)**
The single largest error in the text baseline is fear being classified as
neutral. This is linguistically plausible: fearful Reddit text often reads
as anxious, uncertain, or matter-of-fact ("I'm worried this might happen")
rather than overtly emotional. The GoEmotions label set contains
`nervousness` (mapped here to fear), which overlaps heavily in surface
vocabulary with neutral statements. A fine-tuned model with explicit
five-class supervision would be expected to reduce this confusion.

**Text model — anger → neutral (28 %)**
Angry Reddit text frequently involves passive-aggressive or ironic phrasing
that the zero-shot model reads as neutral. This is a known limitation of
zero-shot inference over coarse mood mappings.

**Audio model — anger → joy (22 % of anger examples)**
This is the audio model's most notable error. In RAVDESS, high-arousal acted
anger and excited joy share similar prosodic features: fast speech rate, high
energy, elevated pitch. A model trained on acted data may learn arousal as a
proxy for emotion and confuse the two high-arousal categories.

**Audio model — sadness → neutral (12.5 %)**
Low-arousal acted sadness and neutral speech are spectrally similar;
this confusion is expected and is common in the speech emotion recognition
literature.

These error patterns suggest complementary weaknesses across modalities:
text struggles with low-arousal ambiguity (fear, anger→neutral), audio
struggles with same-arousal confusion (anger↔joy, sadness↔neutral). This
is consistent with the modest but real gain seen from late fusion.
──────────────────
"""


"""
──────────────────
CELL E  (Code) — confidence calibration check
──────────────────
"""
# ── 17.4  Confidence calibration: are high-confidence predictions more accurate?
pred_df = text_results["predictions_df"].copy()
pred_df["correct"] = pred_df["true_mood"] == pred_df["pred_mood"]

bins = pd.cut(pred_df["confidence"], bins=[0, 0.4, 0.55, 0.7, 0.85, 1.01],
              labels=["<0.40", "0.40–0.55", "0.55–0.70", "0.70–0.85", ">0.85"])
cal = (pred_df.groupby(bins, observed=True)["correct"]
       .agg(accuracy="mean", n="count")
       .round(3))
print("── Text model confidence calibration ────────────────────────────────")
print(cal.to_string())

# Bar chart
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(range(len(cal)), cal["accuracy"], color="#4C72B0", alpha=0.85)
ax.set_xticks(range(len(cal)))
ax.set_xticklabels(cal.index, fontsize=10)
ax.set_xlabel("Confidence bin")
ax.set_ylabel("Accuracy")
ax.set_title("Text Model: Confidence vs Accuracy (calibration)")
ax.axhline(0.2, color="grey", linestyle="--", linewidth=0.8,
           label="Random baseline (5 classes)")
ax.legend()
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(PROJECT_DIR / "report_figures" / "text_confidence_calibration.png", dpi=150)
plt.show()
print("Saved: text_confidence_calibration.png")
