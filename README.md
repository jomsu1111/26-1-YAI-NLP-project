# Hallucination Detection via Feature-Based XGBoost

HaluEval 데이터셋을 기반으로 5가지 언어적 피처를 추출하고, XGBoost 분류기를 학습해 LLM의 hallucination을 탐지하는 파이프라인입니다.

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
│   ├── step3_gpt_factuality.py       # GPT-4o-mini로 factuality score 추출 (비동기 병렬)
│   ├── step4_split.py                # Train / Val / Test 분할
│   ├── step5_train_xgboost.py        # XGBoost 학습 + Optuna 하이퍼파라미터 튜닝
│   ├── step6_evaluate.py             # 최종 모델 평가 및 리포트 생성
│   ├── step7_analysis.py             # SHAP / Calibration / ROC 시각화
│   └── step8_error_analysis.py       # 오답 유형(FP/FN) 분석 + Dialogue 오류 패턴
│
├── data/
│   ├── unified_dataset.csv           # 3개 도메인 통합 (30,000행)
│   ├── feature_matrix.csv            # 피처 추출 완료 (30,000행 × 11컬럼)
│   ├── test_predictions.csv          # v4 테스트셋 예측 결과
│   └── test_predictions_v3.csv       # v3 테스트셋 예측 결과 (비교용)
│
├── splits/
│   ├── train.csv                     # 학습셋 (70%)
│   ├── val.csv                       # 검증셋 (15%)
│   ├── test.csv                      # 테스트셋 (15%)
│   └── scale_pos_weight.txt          # XGBoost 클래스 불균형 보정값
│
├── models/
│   ├── xgb_model_v1_ner.pkl          # v1: NLI + NER + SBERT
│   ├── xgb_model_v2_rougel.pkl       # v2: NLI + ROUGE-L + SBERT
│   ├── xgb_model_v3_all4.pkl         # v3: NLI + NER + SBERT + ROUGE-L
│   └── xgb_model_v4_gpt.pkl          # v4: v3 + GPT Factuality (최종 사용)
│
├── results/
│   ├── evaluation_report.txt                  # 전체 및 도메인별 평가 지표 (v4)
│   ├── feature_comparison.csv                 # 피처 조합별 성능 비교 (v1~v4)
│   ├── optuna_results_v1_ner.csv
│   ├── optuna_results_v2_rougel.csv
│   ├── optuna_results_v3_all4.csv
│   ├── optuna_results_v4_gpt.csv
│   └── error_analysis_dialogue_samples.csv    # Dialogue FP/FN 오답 예시 (각 20개)
│
└── figures/
    ├── shap_summary.png              # SHAP Beeswarm Plot (v4)
    ├── shap_bar.png                  # Feature Importance (Mean |SHAP|) (v4)
    ├── calibration_curve.png         # Calibration Curve (v4)
    ├── domain_roc.png                # 도메인별 ROC / AUROC (v4)
    ├── error_fp_fn_by_domain.png     # 도메인별 FP / FN (v4)
    ├── error_feature_dist.png        # FP / FN 피처값 분포 (v4)
    ├── error_dialogue_features.png   # Dialogue 오류 유형별 피처 분포 (v4)
    ├── feature_comparison.png        # v1~v4 AUROC / F1 비교 차트
    └── v3/                           # v3 모델 결과 figures (비교용)
        ├── shap_bar_v3.png
        ├── shap_summary_v3.png
        ├── calibration_curve_v3.png
        ├── domain_roc_v3.png
        ├── error_fp_fn_by_domain_v3.png
        ├── error_feature_dist_v3.png
        └── error_dialogue_features_v3.png
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
| `gpt_factuality` | `gpt-4o-mini` | context 기반 response의 사실성 점수 (0.0~1.0). GPT가 직접 평가 |

> **피처 추출 소요 시간**: NLI+NER+SBERT+ROUGE-L 약 6~7시간 / GPT Factuality 약 $2.5 (gpt-4o-mini 기준)

---

## 실행 순서

스크립트는 **`scripts/` 폴더에서** 실행합니다.

```bash
# 1. 데이터 로드 및 구조 확인
python step1_load_and_inspect.py

# 2. 3개 도메인 포맷 통일
python step2_format_unify.py

# 3. 피처 추출 (약 6~7시간)
python step3_feature_extraction.py

# 3-1. GPT Factuality Score 추출 (API 키 필요)
export OPENAI_API_KEY="sk-..."
python step3_gpt_factuality.py

# 4. Train / Val / Test 분할
python step4_split.py

# 5. XGBoost 학습 + Optuna 튜닝 (v1~v4 비교)
python step5_train_xgboost.py

# 6. 최종 모델 평가
python step6_evaluate.py

# 7. SHAP / Calibration / ROC 시각화
python step7_analysis.py

# 8. 오답 유형 분석 (FP/FN)
python step8_error_analysis.py
```

---

## 모델 비교 (Val Set 기준)

| 버전 | 피처 조합 | AUROC | F1 |
|------|---------|-------|-----|
| v1 | NLI + NER + SBERT | 0.7218 | 0.6655 |
| v2 | NLI + ROUGE-L + SBERT | 0.8159 | 0.7442 |
| v3 | NLI + NER + SBERT + ROUGE-L | 0.8279 | 0.7529 |
| **v4** | **v3 + GPT Factuality** | **0.9119** | **0.8291** |

→ **v4** 가 AUROC / F1 모두 최고 성능으로 최종 모델로 채택

---

## 최종 성능 (Test Set, v4 모델)

### 전체 지표

| 지표 | v3 | v4 |
|------|----|----|
| AUROC | 0.8459 | **0.9092** |
| F1 | 0.7783 | **0.8286** |
| Precision | 0.7031 | 0.8180 |
| Recall | 0.8715 | 0.8394 |
| Accuracy | - | **0.83** |
| 오답률 | 24.9% | **17.4%** |

### 도메인별 AUROC / 오답률 (v4)

| 도메인 | AUROC | 오답률 | FP | FN |
|--------|-------|--------|----|----|
| QA | **0.9960** | 6.6% | 4 (0.3%) | 94 (6.4%) |
| Dialogue | 0.8712 | 20.2% | 198 (13.2%) | 105 (7.0%) |
| Summarization | 0.8369 | 25.1% | 219 (14.4%) | 163 (10.7%) |

### Feature Importance (v4)

| 피처 | XGBoost Importance | Mean \|SHAP\| |
|------|-------------------|--------------|
| `gpt_factuality` | **0.7257** | **1.4410** |
| `rouge_l` | 0.1610 | 1.2303 |
| `sbert_cosine` | 0.0618 | 0.4780 |
| `nli_score` | 0.0325 | 0.2555 |
| `ner_jaccard` | 0.0190 | 0.1848 |

---

## 주요 결과 해석

- **gpt_factuality**가 XGBoost importance 69.4%로 압도적 1위. GPT가 직접 판단한 사실성 점수가 hallucination 탐지에 가장 강력한 시그널
- **rouge_l**은 SHAP 기준 gpt_factuality와 근접한 영향력(1.23 vs 1.44)으로, GPT 없이도 유용한 피처
- **QA 도메인**은 오답률 6%로 매우 우수, **Summarization**은 24.7%로 가장 어려운 도메인 (context가 길고 복잡)
- **Dialogue**는 FP(15.7%)가 많아 정상 응답을 환각으로 오판하는 경향

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
openai
torch
pandas
numpy
matplotlib
datasets
```

```bash
pip install xgboost optuna shap scikit-learn sentence-transformers transformers \
            rouge-score openai torch pandas numpy matplotlib datasets
python -m spacy download en_core_web_trf
```
