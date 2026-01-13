# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""
测试自定义RM模型加载
验证MyRMForTokenClassification能否通过AutoModelForTokenClassification正确加载
"""

import argparse
import sys

import torch


def test_rm_loading(model_path: str, device: str = "cpu"):
    """
    测试RM模型加载和推理

    Args:
        model_path: RM模型路径
        device: 运行设备 (cpu/cuda)
    """
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    print(f"=" * 60)
    print(f"Testing RM Loading: {model_path}")
    print(f"=" * 60)

    # 1. 加载配置
    print("\n[1/4] Loading config...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"  Config type: {type(config).__name__}")
    print(f"  Model type: {config.model_type}")
    print(f"  Hidden dim str: {getattr(config, 'hidden_dim_str', 'N/A')}")
    print(f"  Num labels: {getattr(config, 'num_labels', 'N/A')}")

    # 验证配置
    assert config.model_type == "myrm", f"Expected model_type='myrm', got '{config.model_type}'"
    assert hasattr(config, "hidden_dim_str"), "Config missing 'hidden_dim_str'"
    print("  [OK] Config validation passed")

    # 2. 加载tokenizer
    print("\n[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  [OK] Tokenizer loaded")

    # 3. 加载模型
    print("\n[3/4] Loading model...")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    print(f"  Model type: {type(model).__name__}")
    print(f"  Device: {device}")
    print(f"  [OK] Model loaded")

    # 验证模型结构
    assert hasattr(model, "myscore"), "Model missing 'myscore' attribute (MLP head)"
    print(f"  MLP head layers: {len(model.myscore.layers)}")

    # 4. 测试推理
    print("\n[4/4] Testing inference...")
    test_prompts = [
        "How can I help someone who is feeling depressed?",
        "What is the capital of France?",
    ]

    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        print(f"\n  Prompt {i+1}: '{prompt[:50]}...'")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Reward score: {logits.squeeze().item():.4f}")

        # 验证输出shape
        assert logits.shape[-1] == 1, f"Expected logits shape [..., 1], got {logits.shape}"

    print("\n" + "=" * 60)
    print("[SUCCESS] All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RM model loading")
    parser.add_argument(
        "--model_path",
        type=str,
        default="merge/merged_model/Naive-RM-saferlhf",
        help="Path to RM model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    args = parser.parse_args()

    try:
        test_rm_loading(args.model_path, args.device)
    except Exception as e:
        print(f"\n[FAILED] Error: {e}")
        sys.exit(1)
