"""
RAGTruth span 레이블 예시 출력
발표용으로 환각 span이 response 어디에 있는지 시각화
"""

from datasets import load_dataset

# HuggingFace에서 RAGTruth 로드
print("RAGTruth 로딩 중...")
ds = load_dataset("leobianco/ragtruth", split="train")

# span 레이블이 있는 샘플만 필터링
samples_with_span = [x for x in ds if x.get("labels") and len(x["labels"]) > 0]
print(f"span 레이블 있는 샘플: {len(samples_with_span)}개\n")

# 첫 번째 예시 출력
sample = samples_with_span[0]
response = sample["response"]
labels   = sample["labels"]

print("=" * 60)
print("RAGTruth Span 레이블 예시")
print("=" * 60)
print(f"\n[Response]\n{response}\n")

print("[Span Labels]")
for lb in labels:
    start      = lb["start"]
    end        = lb["end"]
    text       = lb["text"]
    label_type = lb["label_type"]
    print(f"  start={start}, end={end}")
    print(f"  text: \"{text}\"")
    print(f"  label_type: {label_type}")
    print()

# 하이라이트 버전 출력 (터미널 컬러)
print("[하이라이트 버전]")
result = response
offset = 0
for lb in sorted(labels, key=lambda x: x["start"]):
    s = lb["start"] + offset
    e = lb["end"]   + offset
    highlighted = f"\033[41m\033[97m{result[s:e]}\033[0m"  # 빨간 배경
    result = result[:s] + highlighted + result[e:]
    offset += len(highlighted) - (e - s)

print(result)
