"""
Step 8: XGBoost 평가 및 결과 저장

입력 : xgb_model.pkl, splits/val.csv, splits/test.csv
출력 : test_predictions.csv  (C팀 전달용)
       evaluation_report.txt  (평가 지표 요약)

평가 지표
- 전체  : AUROC (메인), F1, Precision, Recall
- 도메인별: AUROC (qa / summarization / dialogue / data-to-text)

Threshold: val set F1 최대화 기준으로 자동 결정
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

MODEL_PATH   = "../models/xgb_model_v3_all4.pkl"
VAL_PATH     = "../splits/val.csv"
TEST_PATH    = "../splits/test.csv"
PRED_PATH    = "../data/test_predictions.csv"
REPORT_PATH  = "../results/evaluation_report.txt"

FEATURES = ["nli_score", "ner_jaccard", "sbert_cosine", "rouge_l"]


# ============================================================
# 로드
# ============================================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

val  = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

print(f"val set 로드 : {len(val):,} rows")
print(f"test set 로드: {len(test):,} rows")


# ============================================================
# val set으로 최적 threshold 탐색 (F1 최대화)
# ============================================================

X_val  = val[FEATURES].values
y_val  = val["label"].values
val_prob = model.predict_proba(X_val)[:, 1]

thresholds   = np.arange(0.05, 0.95, 0.01)
f1_scores    = [f1_score(y_val, (val_prob >= t).astype(int)) for t in thresholds]
best_idx     = int(np.argmax(f1_scores))
THRESHOLD    = thresholds[best_idx]
best_val_f1  = f1_scores[best_idx]

print(f"\n[Threshold 탐색 결과]")
print(f"  최적 threshold : {THRESHOLD:.2f}")
print(f"  val F1 (0.5)   : {f1_score(y_val, (val_prob >= 0.5).astype(int)):.4f}")
print(f"  val F1 (최적)  : {best_val_f1:.4f}")


# ============================================================
# test set 예측
# ============================================================

X_test = test[FEATURES].values
y_true = test["label"].values
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)


# ============================================================
# 전체 평가 지표
# ============================================================

auroc     = roc_auc_score(y_true, y_prob)
f1        = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)

lines = []
lines.append("=" * 60)
lines.append("XGBoost 평가 결과")
lines.append("=" * 60)
lines.append(f"\n[Threshold: {THRESHOLD:.2f}  (val F1 최대화 기준)]")
lines.append(f"\n[전체 (n={len(test):,})]")
lines.append(f"  AUROC    : {auroc:.4f}")
lines.append(f"  F1       : {f1:.4f}")
lines.append(f"  Precision: {precision:.4f}")
lines.append(f"  Recall   : {recall:.4f}")
lines.append("")
lines.append(classification_report(y_true, y_pred, target_names=["정상(0)", "환각(1)"]))


# ============================================================
# 도메인별 AUROC
# ============================================================

domains = test["domain"].unique()
lines.append("\n[도메인별 AUROC]")

domain_results = {}
for domain in sorted(domains):
    mask = test["domain"] == domain
    n    = mask.sum()
    if n == 0:
        continue
    y_t = y_true[mask]
    y_p = y_prob[mask]

    # 도메인 내 클래스가 1종류인 경우 AUROC 계산 불가
    if len(np.unique(y_t)) < 2:
        lines.append(f"  {domain:<18}: n={n:>5,}  AUROC=N/A (단일 클래스)")
        domain_results[domain] = None
        continue

    d_auroc = roc_auc_score(y_t, y_p)
    domain_results[domain] = d_auroc
    lines.append(f"  {domain:<18}: n={n:>5,}  AUROC={d_auroc:.4f}")

lines.append("")

# 소스별 요약
lines.append("[소스별 AUROC]")
for source in sorted(test["source"].unique()):
    mask = test["source"] == source
    n    = mask.sum()
    y_t  = y_true[mask]
    y_p  = y_prob[mask]
    if len(np.unique(y_t)) < 2:
        lines.append(f"  {source:<12}: n={n:>5,}  AUROC=N/A")
        continue
    s_auroc = roc_auc_score(y_t, y_p)
    lines.append(f"  {source:<12}: n={n:>5,}  AUROC={s_auroc:.4f}")

# ============================================================
# Feature Importance
# ============================================================

lines.append("\n[Feature Importance]")
importances = model.feature_importances_
for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    lines.append(f"  {feat:<18}: {imp:.4f}  {bar}")

report_str = "\n".join(lines)
print(report_str)

# 리포트 저장
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_str)
print(f"\n평가 리포트 저장: {REPORT_PATH}")


# ============================================================
# C팀 전달용 예측 결과 저장
# ============================================================

df_pred = test[["label", "domain", "source"]].copy()
df_pred["predicted_proba"] = y_prob
df_pred["predicted_label"] = y_pred

# feature 값도 포함 (SHAP 분석에 유용)
for feat in FEATURES:
    df_pred[feat] = test[feat].values

df_pred.to_csv(PRED_PATH, index=False)
print(f"예측 결과 저장: {PRED_PATH}")
print(f"컬럼: {df_pred.columns.tolist()}")

print("\n[C팀 전달 파일]")
print(f"  모델  : models/xgb_model_v3_all4.pkl")
print(f"  예측값: data/test_predictions.csv")
