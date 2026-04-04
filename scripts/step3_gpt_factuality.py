"""
Step 3-2: GPT API를 이용한 Factuality Score 추출

입력 : data/feature_matrix.csv
출력 : data/feature_matrix.csv (gpt_factuality 컬럼 추가)

방법
- GPT에게 context + question + response를 주고
  "이 response가 context에 근거해 얼마나 사실인가"를 0.0~1.0으로 평가
- JSON 형식으로 score 반환
- 500행마다 중간 저장 (재시작 시 이어서 계산)
- 비동기 병렬 처리 (CONCURRENCY 동시 요청)

비용 추정 (gpt-4o-mini 기준)
- 행당 약 500 input tokens + 10 output tokens
- 30,000행 기준 약 $2.5
- 모델 변경: OPENAI_MODEL 상수 수정

사전 준비
- pip install openai
- export OPENAI_API_KEY="sk-..."
"""

import os
import time
import json
import asyncio
import pandas as pd
from pathlib import Path
from openai import AsyncOpenAI

BASE        = Path(__file__).parent.parent
INPUT_PATH  = BASE / "data/feature_matrix.csv"
OUTPUT_PATH = BASE / "data/feature_matrix.csv"

OPENAI_MODEL     = "gpt-4o-mini"
CHECKPOINT_EVERY = 500
MAX_RETRIES      = 3
RETRY_DELAY      = 5   # seconds
CONCURRENCY      = 20  # 동시 요청 수

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ============================================================
# 프롬프트 설계
# ============================================================

SYSTEM_PROMPT = """You are a factuality evaluator.
Given a context, an optional question, and a response,
evaluate how factually grounded the response is based solely on the context.

Return a JSON object with:
- "score": float between 0.0 and 1.0
    1.0 = response is fully supported by the context
    0.5 = partially supported or neutral
    0.0 = response contradicts or fabricates information not in the context

Respond ONLY with valid JSON. Example:
{"score": 0.9}"""


def sanitize(text: str) -> str:
    return text.encode("utf-8", errors="ignore").decode("utf-8").replace("\x00", "")

def build_user_prompt(context: str, question: str, response: str) -> str:
    context  = sanitize(context)
    question = sanitize(question)
    response = sanitize(response)
    q_line = f"Question: {question}\n" if question.strip() else ""
    return f"""Context: {context}

{q_line}Response: {response}

Evaluate the factuality of the response based on the context."""


# ============================================================
# GPT 호출 (비동기)
# ============================================================

async def get_factuality_score(semaphore: asyncio.Semaphore, idx: int, context: str, question: str, response: str) -> tuple[int, float]:
    user_prompt = build_user_prompt(
        str(context)[:2000],
        str(question),
        str(response)[:500],
    )

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=20,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                parsed  = json.loads(content)
                score   = float(parsed["score"])
                return idx, max(0.0, min(1.0, score))

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"    [재시도 {attempt+1}/{MAX_RETRIES}] idx={idx} {e}")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"    [실패] idx={idx} {e} → -1.0 저장")
                    return idx, -1.0


# ============================================================
# 메인 실행
# ============================================================

async def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"데이터 로드: {len(df):,} rows")

    if "gpt_factuality" in df.columns:
        done_count = df["gpt_factuality"].notna().sum()
        print(f"\n기존 결과 {done_count:,}개 발견 → 이어서 계산")
    else:
        df["gpt_factuality"] = None
        done_count = 0

    remaining_idx = df[df["gpt_factuality"].isna() | (df["gpt_factuality"] == -1.0)].index.tolist()
    print(f"남은 행 수: {len(remaining_idx):,}\n")

    if len(remaining_idx) == 0:
        print("모든 gpt_factuality 계산 완료!")
        return

    estimated_cost = len(remaining_idx) * 550 / 1_000_000 * 0.15
    print(f"예상 비용 (gpt-4o-mini 기준): ${estimated_cost:.2f}")
    print(f"동시 요청 수: {CONCURRENCY}\n")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    completed = 0

    # CHECKPOINT_EVERY 단위로 배치 처리
    for batch_start in range(0, len(remaining_idx), CHECKPOINT_EVERY):
        batch = remaining_idx[batch_start:batch_start + CHECKPOINT_EVERY]

        tasks = [
            get_factuality_score(semaphore, idx, df.loc[idx, "context"], df.loc[idx, "question"], df.loc[idx, "response"])
            for idx in batch
        ]
        results = await asyncio.gather(*tasks)

        for idx, score in results:
            df.at[idx, "gpt_factuality"] = score

        completed += len(batch)
        elapsed = time.time() - t0
        remaining_est = elapsed / completed * (len(remaining_idx) - completed)
        print(f"  [{completed:>6}/{len(remaining_idx)}] "
              f"경과: {elapsed/60:.1f}분  "
              f"예상 잔여: {remaining_est/60:.1f}분")

        df.to_csv(OUTPUT_PATH, index=False)
        print(f"  → 체크포인트 저장: {len(df):,}행")

    print("\n" + "=" * 50)
    print(f"완료: {OUTPUT_PATH}")
    valid = df["gpt_factuality"].replace(-1.0, None).dropna()
    print(f"\ngpt_factuality 통계 (실패 제외):")
    print(valid.describe().round(4))
    print(f"실패(API 오류): {(df['gpt_factuality'] == -1.0).sum()}건")


if __name__ == "__main__":
    asyncio.run(main())
