"""
Step 6: train / val / test 분할

입력 : feature_matrix.csv
출력 : splits/train.csv, splits/val.csv, splits/test.csv

분할 방법
- HaluEval과 RAGTruth 각각 독립적으로 stratified split (label 기준)
- 비율: train 70% / val 15% / test 15%
- random seed 42 고정
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH  = "../data/feature_matrix.csv"
OUTPUT_DIR  = "../splits"
RANDOM_SEED = 42

FEATURES = ["nli_score", "ner_jaccard", "sbert_cosine"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def split_dataset(df: pd.DataFrame, name: str) -> tuple:
    """
    stratified split: label 비율 유지
    반환: (train_df, val_df, test_df)
    """
    # 1차: train 70% / temp 30%
    train, temp = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    # 2차: temp를 val 50% / test 50% → 전체 대비 15% / 15%
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp["label"],
    )
    print(f"\n[{name}]")
    print(f"  전체 : {len(df):>6,}  (hall rate: {df['label'].mean():.3f})")
    print(f"  train: {len(train):>6,}  (hall rate: {train['label'].mean():.3f})")
    print(f"  val  : {len(val):>6,}  (hall rate: {val['label'].mean():.3f})")
    print(f"  test : {len(test):>6,}  (hall rate: {test['label'].mean():.3f})")
    return train, val, test


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"feature_matrix 로드: {len(df):,} rows")

    # Feature 결측 확인
    missing = df[FEATURES].isnull().sum()
    if missing.any():
        print("\n[경고] feature 결측값 발견:")
        print(missing[missing > 0])
        before = len(df)
        df = df.dropna(subset=FEATURES).reset_index(drop=True)
        print(f"결측 제거 후: {len(df):,} rows (제거: {before - len(df)})")

    # 소스별 분할
    df_halu = df[df["source"] == "halueval"].reset_index(drop=True)
    df_ragt = df[df["source"] == "ragtruth"].reset_index(drop=True)

    all_trains, all_vals, all_tests = [], [], []

    if len(df_halu) > 0:
        tr, va, te = split_dataset(df_halu, "HaluEval")
        all_trains.append(tr); all_vals.append(va); all_tests.append(te)

    if len(df_ragt) > 0:
        tr, va, te = split_dataset(df_ragt, "RAGTruth")
        all_trains.append(tr); all_vals.append(va); all_tests.append(te)

    # 합산 및 셔플
    train = pd.concat(all_trains, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    val   = pd.concat(all_vals,   ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    test  = pd.concat(all_tests,  ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)

    # 저장
    train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    val.to_csv(  f"{OUTPUT_DIR}/val.csv",   index=False)
    test.to_csv( f"{OUTPUT_DIR}/test.csv",  index=False)

    print("\n" + "=" * 50)
    print("최종 분할 결과")
    print(f"  train: {len(train):,}")
    print(f"  val  : {len(val):,}")
    print(f"  test : {len(test):,}")
    print(f"\n저장 위치: {OUTPUT_DIR}/")

    # scale_pos_weight 계산 (XGBoost 학습에 필요)
    n_neg = (train["label"] == 0).sum()
    n_pos = (train["label"] == 1).sum()
    spw   = n_neg / n_pos
    print(f"\nscale_pos_weight (train set 기준): {spw:.4f}")
    print(f"  정상(0): {n_neg:,}  /  환각(1): {n_pos:,}")

    # 저장 (다음 스텝에서 읽어서 사용)
    with open(f"{OUTPUT_DIR}/scale_pos_weight.txt", "w") as f:
        f.write(str(spw))


if __name__ == "__main__":
    main()
