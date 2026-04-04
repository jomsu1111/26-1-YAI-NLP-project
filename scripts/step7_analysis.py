"""
Step 7: SHAP 분석 + Calibration Curve + 도메인별 AUROC 시각화

입력 : xgb_model_v3_all4.pkl, test_predictions.csv
출력 : figures/shap_summary.png
       figures/shap_bar.png
       figures/calibration_curve.png
       figures/domain_roc.png
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import shap
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc

BASE       = Path(__file__).parent.parent
MODEL_PATH = BASE / "models/xgb_model_v4_gpt.pkl"
PRED_PATH  = BASE / "data/test_predictions.csv"
FIG_DIR    = BASE / "figures"
FEATURES   = ["nli_score", "ner_jaccard", "sbert_cosine", "rouge_l", "gpt_factuality"]

os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "figure.dpi"    : 150,
    "font.size"     : 11,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})

# ============================================================
# 데이터 로드
# ============================================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

df = pd.read_csv(PRED_PATH)
X_test = df[FEATURES].values
y_true = df["label"].values
y_prob = df["predicted_proba"].values

print(f"test set 로드: {len(df):,} rows")


# ============================================================
# 1. SHAP 분석
# ============================================================

print("\nSHAP 값 계산 중...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)  # (n_samples, n_features)

# ── 1-1. SHAP Summary Plot (Beeswarm) ────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(
    shap_values, X_test,
    feature_names=FEATURES,
    show=False,
    plot_size=None,
)
plt.title("SHAP Summary Plot", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shap_summary.png")
plt.close()
print(f"  저장: {FIG_DIR}/shap_summary.png")

# ── 1-2. SHAP Bar Plot (평균 절댓값) ─────────────────────
mean_abs_shap = np.abs(shap_values).mean(axis=0)
order = np.argsort(mean_abs_shap)
feat_sorted = [FEATURES[i] for i in order]
vals_sorted = mean_abs_shap[order]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(feat_sorted, vals_sorted, color="#4C72B0", edgecolor="white")
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Feature Importance (SHAP)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shap_bar.png")
plt.close()
print(f"  저장: {FIG_DIR}/shap_bar.png")


# ============================================================
# 2. Calibration Curve
# ============================================================

print("\nCalibration Curve 계산 중...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── 2-1. 전체 ─────────────────────────────────────────────
ax = axes[0]
fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
ax.plot(mean_pred, fraction_pos, "o-", color="#DD4444", lw=2, ms=6, label="XGBoost v3")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curve (Total)", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ── 2-2. 도메인별 ─────────────────────────────────────────
ax = axes[1]
colors  = {"qa": "#2196F3", "summarization": "#FF9800", "dialogue": "#4CAF50"}
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect")

for domain, color in colors.items():
    mask = df["domain"] == domain
    if mask.sum() < 20:
        continue
    fp, mp = calibration_curve(
        y_true[mask], y_prob[mask], n_bins=10, strategy="uniform"
    )
    ax.plot(mp, fp, "o-", color=color, lw=2, ms=5, label=domain)

ax.set_xlabel("Mean Predicted Probability")
ax.set_title("Calibration Curve (Each Domain)", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.suptitle("Calibration Curve", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/calibration_curve.png", bbox_inches="tight")
plt.close()
print(f"  저장: {FIG_DIR}/calibration_curve.png")


# ============================================================
# 3. 도메인별 ROC Curve (AUROC)
# ============================================================

print("\n도메인별 ROC Curve 생성 중...")

domains = sorted(df["domain"].unique())
colors  = {"qa": "#2196F3", "summarization": "#FF9800", "dialogue": "#4CAF50"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── 3-1. 도메인별 ROC 개별 곡선 ───────────────────────────
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

for domain in domains:
    mask  = df["domain"] == domain
    y_t   = y_true[mask]
    y_p   = y_prob[mask]
    if len(np.unique(y_t)) < 2:
        continue
    fpr, tpr, _ = roc_curve(y_t, y_p)
    roc_auc     = auc(fpr, tpr)
    color       = colors.get(domain, "gray")
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{domain}  (AUROC={roc_auc:.4f})")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve by Domain", fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ── 3-2. 도메인별 AUROC 바 차트 ───────────────────────────
ax = axes[1]
domain_aurocs = {}
for domain in domains:
    mask = df["domain"] == domain
    y_t  = y_true[mask]
    y_p  = y_prob[mask]
    if len(np.unique(y_t)) < 2:
        continue
    fpr, tpr, _ = roc_curve(y_t, y_p)
    domain_aurocs[domain] = auc(fpr, tpr)

bar_colors = [colors.get(d, "gray") for d in domain_aurocs]
bars = ax.bar(domain_aurocs.keys(), domain_aurocs.values(),
              color=bar_colors, edgecolor="white", width=0.5)
ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_ylabel("AUROC")
ax.set_title("AUROC by Domain", fontweight="bold")
ax.axhline(0.5, color="gray", lw=1, linestyle="--", alpha=0.5)

plt.suptitle("Domain-wise ROC Analysis", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/domain_roc.png", bbox_inches="tight")
plt.close()
print(f"  저장: {FIG_DIR}/domain_roc.png")


# ============================================================
# 완료
# ============================================================

print("\n" + "=" * 50)
print("생성된 파일:")
for fname in ["shap_summary.png", "shap_bar.png", "calibration_curve.png", "domain_roc.png"]:
    print(f"  {FIG_DIR}/{fname}")
