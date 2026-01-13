# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""
测试预处理后的数据加载
验证parquet数据格式是否符合VERL要求
"""

import argparse
import sys

import numpy as np
import pandas as pd


def test_data_loading(parquet_path: str):
    """
    测试数据加载

    Args:
        parquet_path: parquet文件路径
    """
    print(f"=" * 60)
    print(f"Testing Data Loading: {parquet_path}")
    print(f"=" * 60)

    # 1. 加载数据
    print("\n[1/3] Loading parquet file...")
    df = pd.read_parquet(parquet_path)
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")

    # 2. 验证必需字段
    print("\n[2/3] Validating required fields...")
    required_fields = ["data_source", "prompt", "ability", "reward_model"]

    for field in required_fields:
        assert field in df.columns, f"Missing required field: {field}"
        print(f"  [OK] {field}")

    # 3. 检查数据格式
    print("\n[3/3] Checking data format...")
    sample = df.iloc[0]

    # 检查prompt格式 (parquet会将list转为numpy.ndarray)
    prompt = sample["prompt"]
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    assert isinstance(prompt, list), f"prompt should be list, got {type(prompt)}"
    assert len(prompt) > 0, "prompt should not be empty"
    assert isinstance(prompt[0], dict), f"prompt[0] should be dict, got {type(prompt[0])}"
    assert "role" in prompt[0], "prompt[0] missing 'role'"
    assert "content" in prompt[0], "prompt[0] missing 'content'"
    print(f"  [OK] prompt format: list of messages (stored as ndarray in parquet)")

    # 检查reward_model格式
    rm = sample["reward_model"]
    assert isinstance(rm, dict), f"reward_model should be dict, got {type(rm)}"
    assert "style" in rm, "reward_model missing 'style'"
    print(f"  [OK] reward_model format: {rm}")

    # 打印样例
    print("\n--- Sample Data ---")
    print(f"data_source: {sample['data_source']}")
    print(f"prompt: {sample['prompt']}")
    print(f"ability: {sample['ability']}")
    print(f"reward_model: {sample['reward_model']}")
    if "extra_info" in sample:
        print(f"extra_info: {sample['extra_info']}")

    # 统计信息
    print("\n--- Statistics ---")
    print(f"Unique data_source: {df['data_source'].unique().tolist()}")
    print(f"Unique ability: {df['ability'].unique().tolist()}")

    print("\n" + "=" * 60)
    print("[SUCCESS] Data validation passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test data loading")
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="Data/pku_saferlhf_verl/train.parquet",
        help="Path to parquet file",
    )
    args = parser.parse_args()

    try:
        test_data_loading(args.parquet_path)
    except Exception as e:
        print(f"\n[FAILED] Error: {e}")
        sys.exit(1)
