"""
Step 2 & 3: HaluEval 포맷 통일

입력 : HuggingFace datasets (pminervini/HaluEval)
출력 : unified_dataset.csv
         columns: context, question, response, label, domain, source

각 태스크의 실제 컬럼 구조를 자동 감지해서 변환:
  - 'hallucination' 컬럼이 있으면 → 단일 response + yes/no 레이블
  - 'right_*' / 'hallucinated_*' 컬럼이 있으면 → 쌍(pair) 방식
"""

import pandas as pd
from datasets import load_dataset

OUTPUT_PATH = "../data/unified_dataset.csv"

# 태스크별 컬럼 매핑 정의
# context_col  : 근거 문서 컬럼명
# question_col : 질문 컬럼명 (없으면 None)
# pair         : (정상 응답 컬럼, 환각 응답 컬럼) — pair 방식일 때
# single       : (응답 컬럼, 레이블 컬럼) — single 방식일 때
TASK_CONFIG = {
    "qa": {
        "context_col" : "knowledge",
        "question_col": "question",
        "pair"        : ("right_answer", "hallucinated_answer"),
        "single"      : ("answer", "hallucination"),
        "domain"      : "qa",
    },
    "summarization": {
        "context_col" : "document",
        "question_col": None,
        "pair"        : ("right_summary", "hallucinated_summary"),
        "single"      : ("summary", "hallucination"),
        "domain"      : "summarization",
    },
    "dialogue": {
        "context_col" : "knowledge",
        "question_col": "dialogue_history",
        "pair"        : ("right_response", "hallucinated_response"),
        "single"      : ("response", "hallucination"),
        "domain"      : "dialogue",
    },
}


def convert_halueval_task(ds, task_name: str) -> pd.DataFrame:
    cfg      = TASK_CONFIG[task_name]
    cols     = ds.column_names
    rows     = []

    # 컬럼 구조 자동 감지
    right_col, hall_col = cfg["pair"]
    resp_col,  lbl_col  = cfg["single"]
    is_pair = right_col in cols and hall_col in cols

    print(f"  [{task_name}] 컬럼: {cols}")
    print(f"  [{task_name}] 방식: {'pair' if is_pair else 'single(response+label)'}")

    for item in ds:
        context  = item[cfg["context_col"]]
        question = item[cfg["question_col"]] if cfg["question_col"] else ""

        if is_pair:
            base = {"context": context, "question": question,
                    "domain": cfg["domain"], "source": "halueval"}
            rows.append({**base, "response": item[right_col], "label": 0})
            rows.append({**base, "response": item[hall_col],  "label": 1})
        else:
            rows.append({
                "context" : context,
                "question": question,
                "response": item[resp_col],
                "label"   : 1 if str(item[lbl_col]).strip().lower() == "yes" else 0,
                "domain"  : cfg["domain"],
                "source"  : "halueval",
            })

    return pd.DataFrame(rows)


def main():
    print("HaluEval 로드 중...\n")
    qa_raw  = load_dataset("pminervini/HaluEval", "qa_samples")['data']
    sum_raw = load_dataset("pminervini/HaluEval", "summarization_samples")['data']
    dia_raw = load_dataset("pminervini/HaluEval", "dialogue_samples")['data']

    df_qa  = convert_halueval_task(qa_raw,  "qa")
    df_sum = convert_halueval_task(sum_raw, "summarization")
    df_dia = convert_halueval_task(dia_raw, "dialogue")

    print(f"\n  QA          : {len(df_qa):>6,} rows")
    print(f"  Summarization: {len(df_sum):>6,} rows")
    print(f"  Dialogue    : {len(df_dia):>6,} rows")

    df = pd.concat([df_qa, df_sum, df_dia], ignore_index=True)

    # 결측/빈 문자열 처리
    df["context"]  = df["context"].fillna("").astype(str)
    df["question"] = df["question"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)

    before = len(df)
    df = df[df["response"].str.strip() != ""].reset_index(drop=True)
    if before - len(df):
        print(f"\n빈 response 제거: {before - len(df)}건")

    print("\n" + "=" * 50)
    print(f"최종 데이터셋: {len(df):,} rows")
    print("\n[도메인별]")
    print(df["domain"].value_counts())
    print("\n[레이블 분포]")
    print(df["label"].value_counts())
    print(f"\nHallucination rate: {df['label'].mean():.3f}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n저장 완료: {OUTPUT_PATH}")
    print("컬럼:", df.columns.tolist())


if __name__ == "__main__":
    main()
