"""
Step 7: XGBoost 학습 + Optuna 튜닝 (3가지 feature 조합 비교)

비교 버전:
  v1: NLI + NER + SBERT          (기존 3개)
  v2: NLI + ROUGE-L + SBERT      (NER → ROUGE-L 교체)
  v3: NLI + NER + SBERT + ROUGE-L (4개 전부)

출력:
  xgb_model_v1.pkl / v2 / v3
  optuna_results_v1.csv / v2 / v3
  feature_comparison.csv  (3버전 성능 비교 요약)
"""

import pickle
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_PATH  = "../splits/train.csv"
VAL_PATH    = "../splits/val.csv"
SPW_PATH    = "../splits/scale_pos_weight.txt"
RANDOM_SEED = 42
N_TRIALS    = 100

FEATURE_SETS = {
    "v1_ner":    ["nli_score", "ner_jaccard",  "sbert_cosine"],
    "v2_rougel": ["nli_score", "rouge_l",       "sbert_cosine"],
    "v3_all4":   ["nli_score", "ner_jaccard",  "sbert_cosine", "rouge_l"],
}


# ============================================================
# 데이터 로드
# ============================================================

train = pd.read_csv(TRAIN_PATH)
val   = pd.read_csv(VAL_PATH)
y_train = train["label"].values
y_val   = val["label"].values

with open(SPW_PATH) as f:
    scale_pos_weight = float(f.read().strip())

print(f"train: {len(train):,}  val: {len(val):,}")
print(f"scale_pos_weight: {scale_pos_weight:.4f}\n")


# ============================================================
# 학습 함수
# ============================================================

def run_experiment(version: str, features: list) -> dict:
    print("=" * 55)
    print(f"[{version}] features: {features}")
    print("=" * 55)

    X_train = train[features].values
    X_val   = val[features].values

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators",      100, 300),
            "max_depth"        : trial.suggest_int("max_depth",          3,   6),
            "learning_rate"    : trial.suggest_float("learning_rate",    0.01, 0.1, log=True),
            "subsample"        : trial.suggest_float("subsample",        0.7,  1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.7,  1.0),
            "objective"        : "binary:logistic",
            "eval_metric"      : "auc",
            "scale_pos_weight" : scale_pos_weight,
            "random_state"     : RANDOM_SEED,
            "verbosity"        : 0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # 최적 파라미터로 최종 학습
    best_params = {
        **study.best_params,
        "objective"        : "binary:logistic",
        "eval_metric"      : "auc",
        "scale_pos_weight" : scale_pos_weight,
        "random_state"     : RANDOM_SEED,
        "verbosity"        : 0,
    }
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # val 평가
    val_prob  = final_model.predict_proba(X_val)[:, 1]
    val_auroc = roc_auc_score(y_val, val_prob)
    val_f1    = f1_score(y_val, (val_prob >= 0.5).astype(int))

    # 모델 저장
    model_path = f"../models/xgb_model_{version}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    # Optuna 결과 저장
    study.trials_dataframe().to_csv(f"../results/optuna_results_{version}.csv", index=False)

    # Feature importance 출력
    importances = final_model.feature_importances_
    print(f"\n  val AUROC: {val_auroc:.4f}  val F1: {val_f1:.4f}")
    print(f"  Feature Importance:")
    for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
        print(f"    {feat}: {imp:.4f}")
    print(f"  모델 저장: {model_path}\n")

    return {
        "version"   : version,
        "features"  : str(features),
        "val_auroc" : round(val_auroc, 4),
        "val_f1"    : round(val_f1, 4),
        "best_params": str(study.best_params),
    }


# ============================================================
# 3가지 버전 실행 및 비교
# ============================================================

results = []
for version, features in FEATURE_SETS.items():
    results.append(run_experiment(version, features))

# 비교 요약 출력
df_comparison = pd.DataFrame(results)[["version", "features", "val_auroc", "val_f1"]]
df_comparison.to_csv("../results/feature_comparison.csv", index=False)

print("\n" + "=" * 55)
print("3가지 버전 비교 (val set 기준)")
print("=" * 55)
print(df_comparison.to_string(index=False))

best_version = df_comparison.loc[df_comparison["val_auroc"].idxmax(), "version"]
print(f"\n→ 최고 성능: {best_version}")
print(f"  해당 모델 파일: xgb_model_{best_version}.pkl")
