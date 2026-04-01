"""
Step 4 & 5: NLI / NER / SBERT / ROUGE-L Feature 추출 → feature_matrix.csv 저장

입력 : unified_dataset.csv
출력 : feature_matrix.csv
         columns: context, question, response, label, domain, source,
                  nli_score, ner_jaccard, sbert_cosine, rouge_l

주의사항
- 전체 계산 시간: 약 6~7시간 (ROUGE-L은 5분 이내로 추가 부담 거의 없음)
- 500개마다 중간 저장 (재시작 시 기존 결과 이어서 계산)
- GPU 사용 가능하면 자동으로 활용
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import spacy
from rouge_score import rouge_scorer as rouge_lib
from transformers import pipeline
from sentence_transformers import SentenceTransformer

INPUT_PATH     = "../data/unified_dataset.csv"
OUTPUT_PATH    = "../data/feature_matrix.csv"
CHECKPOINT_EVERY = 500  # 몇 행마다 중간 저장할지


# ============================================================
# 디바이스 설정
# ============================================================

DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
print(f"디바이스: {DEVICE_STR}")


# ============================================================
# 모델 로드 (최초 1회)
# ============================================================

print("NLI 모델 로드 중... (cross-encoder/nli-deberta-v3-large)")
nli_pipe = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-large",
    device=DEVICE,
    truncation=True,
    max_length=512,
)

print("SpaCy 모델 로드 중... (en_core_web_trf)")
nlp = spacy.load("en_core_web_trf", disable=["parser", "senter"])

print("SBERT 모델 로드 중... (all-mpnet-base-v2)")
sbert = SentenceTransformer("all-mpnet-base-v2", device=DEVICE_STR)

print("모델 로드 완료\n")

_rouge_scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)


# ============================================================
# Feature 계산 함수
# ============================================================

def compute_nli(context: str, response: str) -> float:
    """
    (context → premise, response → hypothesis)
    ENTAILMENT 확률 반환 [0, 1]
    """
    if not context.strip() or not response.strip():
        return 0.0
    result = nli_pipe(
        context,
        text_pair=response,
        truncation=True,
        max_length=512,
    )
    # pipeline 반환: [{"label": "ENTAILMENT"|"NEUTRAL"|"CONTRADICTION", "score": float}]
    # cross-encoder NLI는 단일 라벨 반환
    label = result[0]["label"].upper()
    score = result[0]["score"]
    if label == "ENTAILMENT":
        return float(score)
    elif label == "CONTRADICTION":
        return float(1.0 - score)
    else:  # NEUTRAL
        return float(0.5 * (1.0 - score))


def compute_ner_jaccard(context: str, response: str) -> float:
    """
    SpaCy NER로 엔티티 집합 추출 후 Jaccard 유사도 계산 [0, 1]
    둘 다 비어있으면 1.0, 하나만 비어있으면 0.0
    """
    ents_a = {ent.text.lower() for ent in nlp(context).ents}
    ents_b = {ent.text.lower() for ent in nlp(response).ents}

    if not ents_a and not ents_b:
        return 1.0
    if not ents_a or not ents_b:
        return 0.0

    intersection = len(ents_a & ents_b)
    union        = len(ents_a | ents_b)
    return float(intersection / union)


def compute_rouge_l_precision(context: str, response: str) -> float:
    """
    ROUGE-L Precision: response 단어들이 context에 얼마나 등장하는지 [0, 1]
    precision = LCS(context, response) / len(response)
    context가 없는 내용을 response가 언급하면 낮아짐 → 환각 탐지에 유효
    """
    if not context.strip() or not response.strip():
        return 0.0
    scores = _rouge_scorer.score(context, response)
    return float(scores["rougeL"].precision)


def compute_sbert_cosine(context: str, response: str) -> float:
    """
    SBERT 임베딩 코사인 유사도 [-1, 1]
    normalize_embeddings=True → 내적 = 코사인 유사도
    """
    if not context.strip() or not response.strip():
        return 0.0
    vecs = sbert.encode(
        [context, response],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return float(np.dot(vecs[0], vecs[1]))


# ============================================================
# 배치 처리 (NLI는 개별, SBERT는 배치로 처리)
# ============================================================

def compute_sbert_batch(contexts: list, responses: list, batch_size: int = 64) -> list:
    """SBERT를 배치로 계산해 속도 향상"""
    all_texts  = contexts + responses
    embeddings = sbert.encode(
        all_texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    n = len(contexts)
    ctx_embs  = embeddings[:n]
    resp_embs = embeddings[n:]
    scores = [float(np.dot(c, r)) for c, r in zip(ctx_embs, resp_embs)]
    return scores


# ============================================================
# 메인 실행
# ============================================================

def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"데이터 로드: {len(df):,} rows")

    # ── 이어서 계산 (재시작 지원) ─────────────────────────
    if os.path.exists(OUTPUT_PATH):
        df_done = pd.read_csv(OUTPUT_PATH)
        done_count = len(df_done)
        print(f"기존 결과 {done_count}개 발견 → 이어서 계산")
    else:
        df_done = pd.DataFrame()
        done_count = 0

    remaining = df.iloc[done_count:].reset_index(drop=True)
    print(f"남은 행 수: {len(remaining):,}\n")

    if len(remaining) == 0:
        print("모든 feature 계산 완료!")
        return

    # ── SBERT 전체 배치 계산 (가장 빠름) ──────────────────
    print("SBERT 코사인 유사도 배치 계산 시작...")
    t0 = time.time()
    sbert_scores = compute_sbert_batch(
        remaining["context"].tolist(),
        remaining["response"].tolist(),
        batch_size=64,
    )
    print(f"SBERT 완료: {time.time() - t0:.1f}s\n")

    # ── ROUGE-L 전체 계산 (매우 빠름) ────────────────────
    print("ROUGE-L Precision 계산 시작...")
    t0 = time.time()
    rouge_l_scores = [
        compute_rouge_l_precision(str(row["context"]), str(row["response"]))
        for _, row in remaining.iterrows()
    ]
    print(f"ROUGE-L 완료: {time.time() - t0:.1f}s\n")

    # ── NLI + NER 행별 계산 (느림, 체크포인트 필요) ───────
    nli_scores   = []
    ner_jaccards = []

    t0 = time.time()
    for i, row in remaining.iterrows():
        ctx  = str(row["context"])
        resp = str(row["response"])

        nli_scores.append(compute_nli(ctx, resp))
        ner_jaccards.append(compute_ner_jaccard(ctx, resp))

        # 진행률 출력
        if (i + 1) % 100 == 0:
            elapsed   = time.time() - t0
            remaining_est = elapsed / (i + 1) * (len(remaining) - i - 1)
            print(f"  [{i+1:>6}/{len(remaining)}] "
                  f"경과: {elapsed/60:.1f}분  "
                  f"예상 잔여: {remaining_est/60:.1f}분")

        # 체크포인트 저장
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(remaining):
            chunk = remaining.iloc[:i+1].copy()
            chunk["nli_score"]    = nli_scores
            chunk["ner_jaccard"]  = ner_jaccards
            chunk["sbert_cosine"] = sbert_scores[:i+1]
            chunk["rouge_l"]      = rouge_l_scores[:i+1]

            # 기존 완료분과 합치기
            df_save = pd.concat([df_done, chunk], ignore_index=True)
            df_save.to_csv(OUTPUT_PATH, index=False)
            print(f"  → 체크포인트 저장: {len(df_save):,}행")

    # ── 최종 저장 ─────────────────────────────────────────
    remaining["nli_score"]    = nli_scores
    remaining["ner_jaccard"]  = ner_jaccards
    remaining["sbert_cosine"] = sbert_scores
    remaining["rouge_l"]      = rouge_l_scores

    df_final = pd.concat([df_done, remaining], ignore_index=True)
    df_final.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 50)
    print(f"최종 저장 완료: {OUTPUT_PATH}")
    print(f"총 행 수: {len(df_final):,}")
    print("\nFeature 통계:")
    print(df_final[["nli_score", "ner_jaccard", "sbert_cosine", "rouge_l"]].describe().round(4))


if __name__ == "__main__":
    main()
