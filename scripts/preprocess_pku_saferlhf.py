# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""
PKU-SafeRLHF数据预处理脚本
将PKU-SafeRLHF数据集转换为VERL RL训练所需的parquet格式

用法:
    # 使用全部三个数据集（默认）
    python3 scripts/preprocess_pku_saferlhf.py

    # 只用 Alpaca3-8B
    python3 scripts/preprocess_pku_saferlhf.py --subsets Alpaca3-8B

    # 用两个数据集
    python3 scripts/preprocess_pku_saferlhf.py --subsets Alpaca-7B Alpaca2-7B

    # 打乱数据
    python3 scripts/preprocess_pku_saferlhf.py --shuffle --seed 42

    # 单个数据集 + 打乱
    python3 scripts/preprocess_pku_saferlhf.py --subsets Alpaca3-8B --shuffle

参数:
    --local_dataset_path: PKU-SafeRLHF数据集路径 (默认: Data/PKU-SafeRLHF)
    --local_save_dir: 输出目录 (默认: Data/pku_saferlhf_verl)
    --subsets: 数据子集，可多选 (默认: Alpaca-7B Alpaca2-7B Alpaca3-8B)
    --shuffle: 是否打乱数据
    --seed: 随机种子 (默认: 42)
    --train_ratio: 训练集比例 (默认: 0.9)
"""

import argparse
import json
import os
import random

import pandas as pd
from tqdm.auto import tqdm


def load_jsonl(file_path: str) -> list:
    """加载JSONL文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def process_pku_saferlhf(
    local_dataset_path: str,
    local_save_dir: str,
    subsets: list[str],
    train_ratio: float = 0.9,
    shuffle: bool = False,
    seed: int = 42,
):
    """
    处理PKU-SafeRLHF数据集

    Args:
        local_dataset_path: PKU-SafeRLHF数据集路径
        local_save_dir: 输出目录
        subsets: 要使用的数据子集列表
        train_ratio: 训练集比例
        shuffle: 是否打乱数据
        seed: 随机种子
    """
    data_source = "PKU-SafeRLHF"

    # 加载所有训练数据
    all_train_data = []
    all_test_data = []

    for subset in subsets:
        train_path = os.path.join(local_dataset_path, "data", subset, "train.jsonl")
        test_path = os.path.join(local_dataset_path, "data", subset, "test.jsonl")

        if os.path.exists(train_path):
            train_data = load_jsonl(train_path)
            all_train_data.extend(train_data)
            print(f"Loaded {len(train_data)} samples from {subset}/train.jsonl")

        if os.path.exists(test_path):
            test_data = load_jsonl(test_path)
            all_test_data.extend(test_data)
            print(f"Loaded {len(test_data)} samples from {subset}/test.jsonl")

    print(f"\nTotal train samples: {len(all_train_data)}")
    print(f"Total test samples: {len(all_test_data)}")

    def convert_to_verl_format(examples: list, split: str) -> list:
        """转换为VERL格式"""
        processed = []
        for idx, item in enumerate(tqdm(examples, desc=f"Processing {split}")):
            prompt = item["prompt"]

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "safety",
                "reward_model": {"style": "model", "ground_truth": None},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "is_response_0_safe": item.get("is_response_0_safe"),
                    "is_response_1_safe": item.get("is_response_1_safe"),
                    "safer_response_id": item.get("safer_response_id"),
                    "better_response_id": item.get("better_response_id"),
                },
            }
            processed.append(data)
        return processed

    # 转换数据
    train_processed = convert_to_verl_format(all_train_data, "train")
    test_processed = convert_to_verl_format(all_test_data, "test")

    # 如果没有单独的test集，从train中分割
    if len(test_processed) == 0 and len(train_processed) > 0:
        split_idx = int(len(train_processed) * train_ratio)
        test_processed = train_processed[split_idx:]
        train_processed = train_processed[:split_idx]
        print(f"\nNo test set found, split from train: {len(train_processed)} train, {len(test_processed)} val")

    # 打乱数据
    if shuffle:
        random.seed(seed)
        random.shuffle(train_processed)
        random.shuffle(test_processed)
        print(f"\nShuffled data with seed={seed}")

    # 保存为parquet
    os.makedirs(local_save_dir, exist_ok=True)

    train_df = pd.DataFrame(train_processed)
    val_df = pd.DataFrame(test_processed)

    train_path = os.path.join(local_save_dir, "train.parquet")
    val_path = os.path.join(local_save_dir, "val.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"\nSaved {len(train_df)} train samples to {train_path}")
    print(f"Saved {len(val_df)} val samples to {val_path}")

    # 打印样例
    print("\n--- Sample Data ---")
    sample = train_df.iloc[0]
    print(f"data_source: {sample['data_source']}")
    print(f"prompt: {sample['prompt']}")
    print(f"ability: {sample['ability']}")
    print(f"reward_model: {sample['reward_model']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PKU-SafeRLHF dataset for VERL")
    parser.add_argument(
        "--local_dataset_path",
        type=str,
        default="Data/PKU-SafeRLHF",
        help="Path to PKU-SafeRLHF dataset",
    )
    parser.add_argument(
        "--local_save_dir",
        type=str,
        default="Data/pku_saferlhf_verl",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train/val split ratio if no test set exists",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["Alpaca-7B", "Alpaca2-7B", "Alpaca3-8B"],
        help="Data subsets to use (default: all three)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the data after merging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    args = parser.parse_args()

    process_pku_saferlhf(
        local_dataset_path=args.local_dataset_path,
        local_save_dir=args.local_save_dir,
        subsets=args.subsets,
        train_ratio=args.train_ratio,
        shuffle=args.shuffle,
        seed=args.seed,
    )
