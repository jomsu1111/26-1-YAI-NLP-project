"""
Step 1: HaluEval 로드 및 컬럼 구조 확인
"""

import pandas as pd
from datasets import load_dataset

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 80)


print("=" * 60)
print("HaluEval QA")
print("=" * 60)
halueval_qa_raw = load_dataset("pminervini/HaluEval", "qa_samples")
print("\nColumns:", halueval_qa_raw['data'].column_names)
print("\n첫 번째 행:")
print(pd.DataFrame(halueval_qa_raw['data'][:2]))

print("\n" + "=" * 60)
print("HaluEval Summarization")
print("=" * 60)
halueval_sum_raw = load_dataset("pminervini/HaluEval", "summarization_samples")
print("\nColumns:", halueval_sum_raw['data'].column_names)
print("\n첫 번째 행:")
print(pd.DataFrame(halueval_sum_raw['data'][:2]))

print("\n" + "=" * 60)
print("HaluEval Dialogue")
print("=" * 60)
halueval_dia_raw = load_dataset("pminervini/HaluEval", "dialogue_samples")
print("\nColumns:", halueval_dia_raw['data'].column_names)
print("\n첫 번째 행:")
print(pd.DataFrame(halueval_dia_raw['data'][:2]))
