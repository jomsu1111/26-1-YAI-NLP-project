"""
기존 feature_matrix.csv에 rouge_l 컬럼만 추가

NLI/NER/SBERT는 이미 계산되어 있으므로 건드리지 않음
소요 시간: 약 5분
"""

import pandas as pd
from rouge_score import rouge_scorer as rouge_lib

INPUT_PATH  = "./feature_matrix.csv"
OUTPUT_PATH = "./feature_matrix.csv"  # 덮어쓰기

df = pd.read_csv(INPUT_PATH)
print(f"로드: {len(df):,} rows")
print(f"기존 컬럼: {df.columns.tolist()}")

if "rouge_l" in df.columns:
    print("이미 rouge_l 컬럼이 존재함. 종료.")
else:
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)

    def compute_rouge_l_precision(context: str, response: str) -> float:
        if not str(context).strip() or not str(response).strip():
            return 0.0
        return float(scorer.score(str(context), str(response))["rougeL"].precision)

    print("ROUGE-L Precision 계산 중...")
    df["rouge_l"] = [
        compute_rouge_l_precision(row["context"], row["response"])
        for _, row in df.iterrows()
    ]

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n저장 완료: {OUTPUT_PATH}")
    print(f"컬럼: {df.columns.tolist()}")
    print("\nROUGE-L 통계:")
    print(df["rouge_l"].describe().round(4))
