"""
Step 8: 오답 유형 분석 (Error Analysis)

입력 : data/test_predictions.csv, splits/test.csv
출력 : figures/error_fp_fn_by_domain.png
       figures/error_feature_dist.png
       figures/error_dialogue_features.png
       results/error_analysis_dialogue_samples.csv

분석 항목
  1. 도메인별 FP / FN 비율
  2. FP / FN 케이스의 피처값 분포
  3. Dialogue 도메인 오답의 피처 패턴 및 예시
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter

BASE         = Path(__file__).parent.parent   # Code/
PRED_PATH    = BASE / "data/test_predictions.csv"
TEST_PATH    = BASE / "splits/test.csv"
FIG_DIR      = BASE / "figures"
RESULTS_DIR  = BASE / "results"
FEATURES     = ["nli_score", "ner_jaccard", "sbert_cosine", "rouge_l"]

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 한국어 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update({
    "figure.dpi"       : 150,
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})


# ============================================================
# 데이터 로드 및 병합
# ============================================================

pred = pd.read_csv(PRED_PATH)
test = pd.read_csv(TEST_PATH)

# 행 순서가 동일하므로 index 기준 병합
df = test[["context", "question", "response"]].copy()
df = pd.concat([df, pred], axis=1)

# 오답 유형 분류
#   FP (False Positive) : 실제 0(정상) → 예측 1(환각) — 억울하게 환각 판정
#   FN (False Negative) : 실제 1(환각) → 예측 0(정상) — 환각을 놓침
#   TP / TN : 정답
df["error_type"] = "correct"
df.loc[(df["label"] == 0) & (df["predicted_label"] == 1), "error_type"] = "FP"
df.loc[(df["label"] == 1) & (df["predicted_label"] == 0), "error_type"] = "FN"

print(f"전체 테스트 샘플: {len(df):,}")
print(df["error_type"].value_counts())
print(f"\n오답률: {(df['error_type'] != 'correct').mean():.3f}")


# ============================================================
# 1. 도메인별 FP / FN 비율
# ============================================================

print("\n[1] 도메인별 FP / FN 분석")

domains = sorted(df["domain"].unique())
domain_stats = []
for d in domains:
    sub = df[df["domain"] == d]
    n_total = len(sub)
    n_fp = (sub["error_type"] == "FP").sum()
    n_fn = (sub["error_type"] == "FN").sum()
    n_correct = (sub["error_type"] == "correct").sum()
    domain_stats.append({
        "domain"     : d,
        "n_total"    : n_total,
        "correct"    : n_correct,
        "FP"         : n_fp,
        "FN"         : n_fn,
        "FP_rate"    : n_fp / n_total,
        "FN_rate"    : n_fn / n_total,
        "error_rate" : (n_fp + n_fn) / n_total,
    })
    print(f"  {d:<18} total={n_total}  correct={n_correct}  FP={n_fp}({n_fp/n_total:.2%})  FN={n_fn}({n_fn/n_total:.2%})")

df_stats = pd.DataFrame(domain_stats)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1-1. 도메인별 FP / FN 절대 수
ax = axes[0]
x = np.arange(len(domains))
w = 0.35
bars_fp = ax.bar(x - w/2, df_stats["FP"], width=w, label="FP (정상→환각 오판)", color="#E53935", alpha=0.85)
bars_fn = ax.bar(x + w/2, df_stats["FN"], width=w, label="FN (환각 미탐지)",    color="#1E88E5", alpha=0.85)
ax.bar_label(bars_fp, padding=3, fontsize=9)
ax.bar_label(bars_fn, padding=3, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(domains)
ax.set_ylabel("샘플 수")
ax.set_title("도메인별 FP / FN 수", fontweight="bold")
ax.legend(fontsize=9)

# 1-2. 도메인별 FP / FN 비율 (stacked)
ax = axes[1]
correct_rate = df_stats["correct"] / df_stats["n_total"]
fp_rate      = df_stats["FP_rate"]
fn_rate      = df_stats["FN_rate"]

ax.bar(domains, correct_rate, label="Correct",           color="#43A047", alpha=0.85)
ax.bar(domains, fp_rate,      bottom=correct_rate,       label="FP",  color="#E53935", alpha=0.85)
ax.bar(domains, fn_rate,      bottom=correct_rate+fp_rate, label="FN", color="#1E88E5", alpha=0.85)

for i, d in enumerate(domains):
    ax.text(i, correct_rate[i] + fp_rate[i] + fn_rate[i] + 0.01,
            f"{(fp_rate[i]+fn_rate[i]):.1%}", ha="center", fontsize=9, fontweight="bold")

ax.set_ylim(0, 1.1)
ax.set_ylabel("비율")
ax.set_title("도메인별 오답 비율 (stacked)", fontweight="bold")
ax.legend(fontsize=9, loc="upper right")

plt.suptitle("Domain-wise Error Analysis (FP / FN)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/error_fp_fn_by_domain.png", bbox_inches="tight")
plt.close()
print(f"\n  저장: {FIG_DIR}/error_fp_fn_by_domain.png")


# ============================================================
# 2. FP / FN 케이스의 피처값 분포
# ============================================================

print("\n[2] FP / FN 피처값 분포 분석")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
colors = {"FP": "#E53935", "FN": "#1E88E5", "correct": "#43A047"}
labels = {"FP": "FP (정상→환각 오판)", "FN": "FN (환각 미탐지)", "correct": "Correct"}

for col_idx, feat in enumerate(FEATURES):
    for row_idx, error_type in enumerate(["FP", "FN"]):
        ax = axes[row_idx][col_idx]

        # correct vs error_type
        vals_correct = df[df["error_type"] == "correct"][feat].dropna()
        vals_error   = df[df["error_type"] == error_type][feat].dropna()

        bins = np.linspace(
            min(vals_correct.min(), vals_error.min()),
            max(vals_correct.max(), vals_error.max()),
            30
        )
        ax.hist(vals_correct, bins=bins, alpha=0.5, color="#43A047", label="Correct", density=True)
        ax.hist(vals_error,   bins=bins, alpha=0.7, color=colors[error_type], label=error_type, density=True)

        ax.axvline(vals_correct.median(), color="#43A047", linestyle="--", lw=1.2)
        ax.axvline(vals_error.median(),   color=colors[error_type], linestyle="--", lw=1.2)

        ax.set_title(f"{feat}\n({error_type})", fontsize=9, fontweight="bold")
        ax.set_xlabel(feat, fontsize=8)
        if col_idx == 0:
            ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7)

        # median 값 출력
        print(f"  {feat:18} | {error_type} median={vals_error.median():.3f}  correct median={vals_correct.median():.3f}")

plt.suptitle("Feature Distribution: FP / FN vs Correct", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/error_feature_dist.png", bbox_inches="tight")
plt.close()
print(f"\n  저장: {FIG_DIR}/error_feature_dist.png")


# ============================================================
# 3. Dialogue 도메인 오답 심층 분석
# ============================================================

print("\n[3] Dialogue 도메인 오답 분석")

dia = df[df["domain"] == "dialogue"].copy()
dia_fp = dia[dia["error_type"] == "FP"]
dia_fn = dia[dia["error_type"] == "FN"]
dia_correct = dia[dia["error_type"] == "correct"]

print(f"  Dialogue 전체: {len(dia)}  FP: {len(dia_fp)}  FN: {len(dia_fn)}  Correct: {len(dia_correct)}")

# 3-1. Dialogue FP / FN 피처 분포 비교
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, feat in enumerate(FEATURES):
    ax = axes[i]
    for etype, color, label in [
        ("correct", "#43A047", "Correct"),
        ("FP",      "#E53935", "FP"),
        ("FN",      "#1E88E5", "FN"),
    ]:
        vals = dia[dia["error_type"] == etype][feat].dropna()
        ax.hist(vals, bins=20, alpha=0.55, color=color, label=label, density=True)
        ax.axvline(vals.median(), color=color, linestyle="--", lw=1.5)

    ax.set_title(feat, fontweight="bold", fontsize=10)
    ax.set_xlabel("value", fontsize=9)
    if i == 0:
        ax.set_ylabel("Density")
    ax.legend(fontsize=8)

plt.suptitle("Dialogue Domain — Feature Distribution by Error Type", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/error_dialogue_features.png", bbox_inches="tight")
plt.close()
print(f"  저장: {FIG_DIR}/error_dialogue_features.png")

# 3-2. Dialogue FP 피처 통계 출력
print("\n  [Dialogue FP 피처 통계] — 정상인데 환각으로 오판한 케이스")
print(dia_fp[FEATURES].describe().round(3).to_string())

print("\n  [Dialogue FN 피처 통계] — 환각인데 정상으로 놓친 케이스")
print(dia_fn[FEATURES].describe().round(3).to_string())

# 3-3. Dialogue FN 케이스: response에만 등장하는 단어 빈도 (hallucination 힌트)
print("\n  [Dialogue FN — context에 없는 response 단어 분석]")

def get_novel_words(context: str, response: str) -> list:
    """response에는 있지만 context에는 없는 단어 (소문자, 3글자 이상)"""
    ctx_words  = set(str(context).lower().split())
    resp_words = set(str(response).lower().split())
    novel = [w.strip(".,!?;:'\"") for w in resp_words - ctx_words if len(w) >= 3]
    return [w for w in novel if w.isalpha()]

fn_novel_words = []
for _, row in dia_fn.iterrows():
    fn_novel_words.extend(get_novel_words(row["context"], row["response"]))

# 불용어 간단 제거
STOPWORDS = {"the", "and", "that", "this", "with", "for", "are", "was",
             "have", "has", "been", "not", "you", "what", "but", "they",
             "from", "will", "can", "its", "all", "one", "our", "your",
             "their", "would", "could", "should", "also", "more", "just"}
fn_novel_words = [w for w in fn_novel_words if w not in STOPWORDS]

top_novel = Counter(fn_novel_words).most_common(30)
print("  FN 케이스에서 context에 없는 단어 Top 30:")
for word, cnt in top_novel:
    print(f"    {word:<20} {cnt}회")

# 3-4. FP / FN 예시 저장 (각 20개)
fp_samples = dia_fp[["context", "question", "response", "label", "predicted_label",
                      "predicted_proba"] + FEATURES].head(20)
fn_samples = dia_fn[["context", "question", "response", "label", "predicted_label",
                      "predicted_proba"] + FEATURES].head(20)

fp_samples.insert(0, "error_type", "FP")
fn_samples.insert(0, "error_type", "FN")

samples = pd.concat([fp_samples, fn_samples], ignore_index=True)
out_path = f"{RESULTS_DIR}/error_analysis_dialogue_samples.csv"
samples.to_csv(out_path, index=False)
print(f"\n  Dialogue 오답 예시 저장: {out_path} ({len(samples)}개)")


# ============================================================
# 요약 출력
# ============================================================

print("\n" + "=" * 60)
print("오답 분석 요약")
print("=" * 60)
print(f"\n전체 오답률: {(df['error_type'] != 'correct').mean():.2%}")
print(f"  FP (정상→환각 오판): {(df['error_type']=='FP').sum()}개 ({(df['error_type']=='FP').mean():.2%})")
print(f"  FN (환각 미탐지)   : {(df['error_type']=='FN').sum()}개 ({(df['error_type']=='FN').mean():.2%})")

print("\n[도메인별 오답률 요약]")
for _, row in df_stats.iterrows():
    print(f"  {row['domain']:<18} 오답률={row['error_rate']:.2%}  FP={row['FP_rate']:.2%}  FN={row['FN_rate']:.2%}")

print("\n생성된 파일:")
print(f"  {FIG_DIR}/error_fp_fn_by_domain.png")
print(f"  {FIG_DIR}/error_feature_dist.png")
print(f"  {FIG_DIR}/error_dialogue_features.png")
print(f"  {RESULTS_DIR}/error_analysis_dialogue_samples.csv")
