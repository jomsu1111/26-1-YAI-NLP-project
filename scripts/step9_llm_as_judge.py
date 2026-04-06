"""
Step 9: LLM as Judge 평가

입력 : data/feature_matrix.csv
출력 : data/llm_judge_results.csv   (샘플 + judge 결과)
       results/llm_judge_report.txt  (성능 리포트)

방법
- 도메인별 1,000개 stratified sampling (label 비율 유지)
- GPT에게 context + question + response를 주고
  "이 response가 context에 근거해 사실인가" binary(0/1) 판정

비용 추정 (gpt-4o-mini 기준)
- 행당 약 500 input tokens + 5 output tokens
- 3,000행 기준 약 $0.25
"""

import os
import json
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from openai import AsyncOpenAI
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

BASE         = Path(__file__).parent.parent
INPUT_PATH   = BASE / "data/feature_matrix.csv"
OUTPUT_PATH  = BASE / "data/llm_judge_results.csv"
REPORT_PATH  = BASE / "results/llm_judge_report.txt"

OPENAI_MODEL      = "gpt-5-mini"
SAMPLE_PER_DOMAIN = 1000
CHECKPOINT_EVERY  = 200
MAX_RETRIES       = 3
RETRY_DELAY       = 5
CONCURRENCY       = 20

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ============================================================
# 샘플링
# ============================================================

def stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """도메인별로 label 비율을 유지하면서 n개 샘플링"""
    samples = []
    for domain, group in df.groupby("domain"):
        sampled = group.groupby("label", group_keys=False).apply(
            lambda x: x.sample(
                n=round(n * len(x) / len(group)),
                random_state=seed,
            )
        )
        if len(sampled) > n:
            sampled = sampled.sample(n=n, random_state=seed)
        elif len(sampled) < n:
            remaining = group[~group.index.isin(sampled.index)]
            extra = remaining.sample(n=n - len(sampled), random_state=seed)
            sampled = pd.concat([sampled, extra])
        samples.append(sampled)
        print(f"  {domain:<18}: {len(sampled)}개  (label=0: {(sampled['label']==0).sum()}, label=1: {(sampled['label']==1).sum()})")
    return pd.concat(samples).reset_index(drop=True)


# ============================================================
# 프롬프트 설계
# ============================================================

SYSTEM_PROMPT = """You are a hallucination detector.
Given a context, an optional question, and a response,
determine whether the response contains hallucination.

Return a JSON object with:
- "label": integer, either 0 or 1
    0 = the response is factually supported by the context (no hallucination)
    1 = the response contains hallucination or is not supported by the context

Respond ONLY with valid JSON. Example:
{"label": 0}"""


def sanitize(text: str) -> str:
    return str(text).encode("utf-8", errors="ignore").decode("utf-8").replace("\x00", "")


def build_user_prompt(context: str, question: str, response: str) -> str:
    context  = sanitize(context)[:2000]
    question = sanitize(question)
    response = sanitize(response)[:500]
    q_line = f"Question: {question}\n" if question.strip() else ""
    return f"""Context: {context}

{q_line}Response: {response}

Is this response factually supported by the context?"""


# ============================================================
# GPT 호출 (비동기)
# ============================================================

async def get_judge_label(
    semaphore: asyncio.Semaphore,
    idx: int,
    context: str,
    question: str,
    response: str,
) -> tuple[int, int]:
    user_prompt = build_user_prompt(context, question, response)

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_completion_tokens=1000,
                )
                content = resp.choices[0].message.content.strip()
                # JSON 파싱 시도, 실패 시 텍스트에서 0/1 추출
                try:
                    parsed = json.loads(content)
                    label  = int(parsed["label"])
                except Exception:
                    if "0" in content:
                        label = 0
                    elif "1" in content:
                        label = 1
                    else:
                        raise ValueError(f"파싱 불가: {content!r}")
                return idx, max(0, min(1, label))

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"    [재시도 {attempt+1}/{MAX_RETRIES}] idx={idx} {e}")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"    [실패] idx={idx} {e} → -1 저장")
                    return idx, -1


# ============================================================
# 메인 실행
# ============================================================

async def main():
    # 1. 데이터 로드 및 샘플링
    df_full = pd.read_csv(INPUT_PATH)
    print(f"전체 데이터: {len(df_full):,}행\n")

    print(f"도메인별 {SAMPLE_PER_DOMAIN}개 stratified sampling:")
    df = stratified_sample(df_full, SAMPLE_PER_DOMAIN)
    print(f"\n총 샘플: {len(df):,}행\n")

    # 2. 이어서 계산 지원
    if OUTPUT_PATH.exists():
        df_prev = pd.read_csv(OUTPUT_PATH)
        if "llm_judge" in df_prev.columns and len(df_prev) == len(df):
            df["llm_judge"] = df_prev["llm_judge"].values
            done = df["llm_judge"].notna().sum()
            print(f"기존 결과 {done:,}개 발견 → 이어서 계산\n")
        else:
            df["llm_judge"] = None
    else:
        df["llm_judge"] = None

    remaining_idx = df[df["llm_judge"].isna() | (df["llm_judge"] == -1)].index.tolist()
    print(f"남은 행 수: {len(remaining_idx):,}")

    if remaining_idx:
        estimated_cost = len(remaining_idx) * 520 / 1_000_000 * 0.15
        print(f"예상 비용 (gpt-4o-mini 기준): ${estimated_cost:.2f}")
        print(f"동시 요청 수: {CONCURRENCY}\n")

        semaphore = asyncio.Semaphore(CONCURRENCY)
        completed = 0

        for batch_start in range(0, len(remaining_idx), CHECKPOINT_EVERY):
            batch = remaining_idx[batch_start:batch_start + CHECKPOINT_EVERY]
            tasks = [
                get_judge_label(
                    semaphore, i,
                    df.loc[i, "context"],
                    df.loc[i, "question"],
                    df.loc[i, "response"],
                )
                for i in batch
            ]
            results = await asyncio.gather(*tasks)

            for i, label in results:
                df.at[i, "llm_judge"] = label

            completed += len(batch)
            print(f"  [{completed:>5}/{len(remaining_idx)}] 체크포인트 저장...")
            df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nLLM judge 완료. 저장: {OUTPUT_PATH}")

    # ============================================================
    # 평가
    # ============================================================

    df_eval = df[df["llm_judge"] != -1].copy()
    df_eval["llm_judge"] = df_eval["llm_judge"].astype(int)
    failed = (df["llm_judge"] == -1).sum()
    print(f"\n평가 대상: {len(df_eval):,}행  (API 실패 제외: {failed}건)")

    y_true  = df_eval["label"].values
    y_judge = df_eval["llm_judge"].values

    lines = []
    lines.append("=" * 60)
    lines.append(f"LLM as Judge 평가 결과  ({OPENAI_MODEL})")
    lines.append("=" * 60)
    lines.append(f"\n샘플: 도메인별 {SAMPLE_PER_DOMAIN}개 × 3 = {len(df_eval):,}행\n")
    lines.append(f"  Accuracy : {accuracy_score(y_true, y_judge):.4f}")
    lines.append(f"  F1       : {f1_score(y_true, y_judge):.4f}")
    lines.append(f"  Precision: {precision_score(y_true, y_judge):.4f}")
    lines.append(f"  Recall   : {recall_score(y_true, y_judge):.4f}")
    lines.append("")
    lines.append(classification_report(y_true, y_judge, target_names=["정상(0)", "환각(1)"]))

    lines.append("-" * 60)
    lines.append("[도메인별 Accuracy]")
    for domain in sorted(df_eval["domain"].unique()):
        mask  = df_eval["domain"].values == domain
        acc   = accuracy_score(y_true[mask], y_judge[mask])
        f1    = f1_score(y_true[mask], y_judge[mask])
        lines.append(f"  {domain:<18}: Accuracy={acc:.4f}  F1={f1:.4f}")

    report_str = "\n".join(lines)
    print("\n" + report_str)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"\n리포트 저장: {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
