# Hallucination Detection via Feature-Based XGBoost

HaluEval 데이터셋을 기반으로 4가지 언어적 피처를 추출하고, XGBoost 분류기를 학습해 LLM의 hallucination을 탐지하는 파이프라인입니다.

---

## 프로젝트 구조

```
Code/
├── README.md
│
├── scripts/
│   ├── step1_load_and_inspect.py     # 데이터셋 로드 및 구조 확인
│   ├── step2_format_unify.py         # 3개 도메인 포맷 통일 → unified_dataset.csv
│   ├── step3_feature_extraction.py   # NLI / NER / SBERT / ROUGE-L 피처 추출
│   ├── step3_add_rouge_l.py          # 기존 feature_matrix에 ROUGE-L 컬럼만 추가
│   ├── step4_split.py                # Train / Val / Test 분할
│   ├── step5_train_xgboost.py        # XGBoost 학습 + Optuna 하이퍼파라미터 튜닝
│   ├── step6_evaluate.py             # 최종 모델 평가 및 리포트 생성
│   └── step7_analysis.py             # SHAP / Calibration / ROC 시각화
│
├── data/
│   ├── unified_dataset.csv       # 3개 도메인 통합 (30,000행)
│   ├── feature_matrix.csv        # 피처 추출 완료 (30,000행 × 10컬럼)
│   └── test_predictions.csv      # 테스트셋 예측 결과 (C팀 전달용)
│
├── splits/
│   ├── train.csv                 # 학습셋 (70%)
│   ├── val.csv                   # 검증셋 (15%)
│   ├── test.csv                  # 테스트셋 (15%)
│   └── scale_pos_weight.txt      # XGBoost 클래스 불균형 보정값
│
├── models/
│   ├── xgb_model_v1_ner.pkl      # v1: NLI + NER + SBERT
│   ├── xgb_model_v2_rougel.pkl   # v2: NLI + ROUGE-L + SBERT
│   └── xgb_model_v3_all4.pkl     # v3: NLI + NER + SBERT + ROUGE-L (최종 사용)
│
├── results/
│   ├── evaluation_report.txt         # 전체 및 도메인별 평가 지표
│   ├── feature_comparison.csv        # 3가지 피처 조합 성능 비교
│   ├── optuna_results_v1_ner.csv     # v1 Optuna 튜닝 로그
│   ├── optuna_results_v2_rougel.csv  # v2 Optuna 튜닝 로그
│   └── optuna_results_v3_all4.csv    # v3 Optuna 튜닝 로그
│
└── figures/
    ├── shap_summary.png          # SHAP Beeswarm Plot
    ├── shap_bar.png              # Feature Importance (Mean |SHAP|)
    ├── calibration_curve.png     # Calibration Curve (전체 + 도메인별)
    └── domain_roc.png            # 도메인별 ROC / AUROC
```

---

## 데이터셋

**HaluEval** (`pminervini/HaluEval`, HuggingFace)

| 도메인 | 샘플 수 | 설명 |
|--------|---------|------|
| QA | 10,000 | 질문-답변 기반 hallucination |
| Summarization | 10,000 | 문서 요약 기반 hallucination |
| Dialogue | 10,000 | 대화 응답 기반 hallucination |
| **합계** | **30,000** | hallucination rate 50% (balanced) |

- **Label**: `0` = faithful(정상), `1` = hallucination(환각)
- **분할**: Stratified split (label 비율 유지) — Train 70% / Val 15% / Test 15%

---

## 피처 설명

| 피처 | 모델/방법 | 설명 |
|------|----------|------|
| `nli_score` | `cross-encoder/nli-deberta-v3-large` | context → response 간 ENTAILMENT 확률. 높을수록 faithful |
| `ner_jaccard` | `spaCy en_core_web_trf` | context와 response의 NER 엔티티 집합 Jaccard 유사도 |
| `sbert_cosine` | `all-mpnet-base-v2` | context-response 임베딩 코사인 유사도 |
| `rouge_l` | `rouge_score` (ROUGE-L Precision) | response 단어가 context에 얼마나 등장하는지. 낮으면 hallucination 의심 |

> **피처 추출 소요 시간**: 약 6~7시간 (NLI + NER 병목), ROUGE-L은 약 5분

---

## 실행 순서

스크립트는 **프로젝트 루트(`Code/`)에서** 실행해야 상대경로(`../data/` 등)가 올바르게 동작합니다.

```bash
# 1. 데이터 로드 및 구조 확인
python scripts/step1_load_and_inspect.py

# 2. 3개 도메인 포맷 통일
python scripts/step2_format_unify.py
# → data/unified_dataset.csv 생성

# 3. 피처 추출 (약 6~7시간 소요, 500행마다 체크포인트 저장)
python scripts/step3_feature_extraction.py
# → data/feature_matrix.csv 생성

# 3-1. 이미 NLI/NER/SBERT가 계산된 경우 ROUGE-L만 추가
python scripts/step3_add_rouge_l.py

# 4. Train / Val / Test 분할
python scripts/step4_split.py
# → splits/train.csv, val.csv, test.csv 생성

# 5. XGBoost 학습 + Optuna 튜닝 (3가지 피처 조합 비교)
python scripts/step5_train_xgboost.py
# → models/xgb_model_v{1,2,3}.pkl, results/optuna_results_v{1,2,3}.csv 생성

# 6. 최종 모델 평가
python scripts/step6_evaluate.py
# → results/evaluation_report.txt, data/test_predictions.csv 생성

# 7. SHAP / Calibration / ROC 시각화
python scripts/step7_analysis.py
# → figures/*.png 생성
```

---

## 모델 비교 (Val Set 기준)

| 버전 | 피처 조합 | AUROC | F1 |
|------|---------|-------|-----|
| v1 | NLI + NER + SBERT | 0.7218 | 0.6655 |
| v2 | NLI + ROUGE-L + SBERT | 0.8159 | 0.7442 |
| **v3** | **NLI + NER + SBERT + ROUGE-L** | **0.8279** | **0.7529** |

→ **v3 (4개 피처 전부)** 가 최고 성능으로 최종 모델로 채택

---

## 최종 성능 (Test Set, v3 모델)

### 전체 지표

| 지표 | 값 |
|------|-----|
| AUROC | **0.8194** |
| F1 | 0.7655 |
| Precision | 0.6663 |
| Recall | 0.8993 |
| Threshold | 0.40 (val F1 최대화) |

### 도메인별 AUROC

| 도메인 | AUROC | 해석 |
|--------|-------|------|
| QA | **0.9904** | 매우 우수 |
| Summarization | 0.7281 | 준수 |
| Dialogue | 0.6875 | 개선 필요 |

> Dialogue 도메인은 정답이 모호하고 정형화된 참조 답변이 없어 ROUGE-L / NLI 기반 피처의 효과가 제한됨

### Feature Importance (XGBoost)

| 피처 | Importance | Mean |SHAP| |
|------|-----------|------|
| rouge_l | 0.5644 | 1.0435 |
| sbert_cosine | 0.1919 | 0.4408 |
| nli_score | 0.1387 | 0.3155 |
| ner_jaccard | 0.1050 | 0.2121 |

---

## 주요 결과 해석

- **rouge_l**이 압도적으로 중요한 피처: response가 context에 없는 내용을 포함할수록 hallucination 확률 증가
- **Calibration**: 전체 기준으로 양호하나, QA 도메인에서 과신(overconfident) 경향
- **도메인 불균형**: QA(0.99) vs Dialogue(0.69) — 도메인별 피처 설계 또는 도메인 특화 모델이 필요

---

## 의존 패키지

```
xgboost
optuna
shap
scikit-learn
sentence-transformers
transformers
spacy
rouge-score
torch
pandas
numpy
matplotlib
datasets
```

```bash
pip install xgboost optuna shap scikit-learn sentence-transformers transformers \
            rouge-score torch pandas numpy matplotlib datasets
python -m spacy download en_core_web_trf
```
