"""
Step 10: LLM as Judge 결과 분석

입력 : data/llm_judge_results.csv
출력 : results/llm_judge_report.txt
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

BASE        = Path(__file__).parent.parent
INPUT_PATH  = BASE / "data/llm_judge_results.csv"
REPORT_PATH = BASE / "results/llm_judge_report.txt"

df = pd.read_csv(INPUT_PATH)

# 실패(-1) 제거
df = df[df["llm_judge"] != -1].copy()
df["llm_judge"] = df["llm_judge"].astype(int)

y_true = df["label"].values
y_pred = df["llm_judge"].values

lines = []
lines.append("=" * 60)
lines.append("LLM as Judge 평가 결과")
lines.append("=" * 60)
lines.append(f"\n평가 샘플: {len(df):,}행  (API 실패 제외)\n")

# 전체 지표
lines.append("[전체 성능]")
lines.append(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
lines.append(f"  F1       : {f1_score(y_true, y_pred):.4f}")
lines.append(f"  Precision: {precision_score(y_true, y_pred):.4f}")
lines.append(f"  Recall   : {recall_score(y_true, y_pred):.4f}")
lines.append("")
lines.append(classification_report(y_true, y_pred, target_names=["정상(0)", "환각(1)"]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
lines.append("[Confusion Matrix]")
lines.append(f"  {'':12}  예측=0  예측=1")
lines.append(f"  {'실제=0':12}  {cm[0][0]:>6}  {cm[0][1]:>6}")
lines.append(f"  {'실제=1':12}  {cm[1][0]:>6}  {cm[1][1]:>6}")
lines.append("")

# 도메인별 성능
lines.append("-" * 60)
lines.append("[도메인별 성능]")
lines.append(f"  {'도메인':<18}  {'n':>5}  {'Accuracy':>9}  {'F1':>7}  {'Precision':>9}  {'Recall':>7}")
lines.append(f"  {'-'*18}  {'-'*5}  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*7}")

for domain in sorted(df["domain"].unique()):
    mask = df["domain"] == domain
    yt   = y_true[mask]
    yp   = y_pred[mask]
    lines.append(
        f"  {domain:<18}  {mask.sum():>5}"
        f"  {accuracy_score(yt, yp):>9.4f}"
        f"  {f1_score(yt, yp):>7.4f}"
        f"  {precision_score(yt, yp):>9.4f}"
        f"  {recall_score(yt, yp):>7.4f}"
    )

lines.append("")

# 예측 분포
lines.append("-" * 60)
lines.append("[예측 분포]")
lines.append(f"  judge=0 (정상 판정): {(y_pred==0).sum():,}개  ({(y_pred==0).mean()*100:.1f}%)")
lines.append(f"  judge=1 (환각 판정): {(y_pred==1).sum():,}개  ({(y_pred==1).mean()*100:.1f}%)")
lines.append("")
lines.append("[실제 label 분포]")
lines.append(f"  label=0 (정상): {(y_true==0).sum():,}개  ({(y_true==0).mean()*100:.1f}%)")
lines.append(f"  label=1 (환각): {(y_true==1).sum():,}개  ({(y_true==1).mean()*100:.1f}%)")

report_str = "\n".join(lines)
print(report_str)

REPORT_PATH.parent.mkdir(exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_str)
print(f"\n리포트 저장: {REPORT_PATH}")
