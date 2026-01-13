# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""
GRPO Group Logger
保存每个group内的n条消息(prompt+response)及对应的RM reward
"""

import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch


class GRPOGroupLogger:
    """GRPO训练过程中的group logging工具"""

    def __init__(self, log_dir: str, log_freq: int = 10):
        """
        Args:
            log_dir: 日志保存目录
            log_freq: 记录频率（每多少step记录一次）
        """
        self.log_dir = log_dir
        self.log_freq = log_freq
        os.makedirs(log_dir, exist_ok=True)

    def should_log(self, global_step: int) -> bool:
        """判断当前step是否需要记录"""
        return global_step % self.log_freq == 0

    def log_groups(
        self,
        global_step: int,
        prompts: torch.Tensor,
        responses: torch.Tensor,
        rewards: torch.Tensor,
        uids: np.ndarray,
        tokenizer,
        max_groups: Optional[int] = None,
    ):
        """
        记录group数据到JSON文件

        Args:
            global_step: 当前训练步数
            prompts: prompt tensor [bs*n, prompt_len]
            responses: response tensor [bs*n, response_len]
            rewards: reward tensor [bs*n] 或 [bs*n, response_len]
            uids: uid数组 [bs*n]，用于分组
            tokenizer: 用于解码的tokenizer
            max_groups: 最大记录的group数量（None表示全部记录）
        """
        # 解码prompts和responses
        prompt_texts = tokenizer.batch_decode(prompts, skip_special_tokens=True)
        response_texts = tokenizer.batch_decode(responses, skip_special_tokens=True)

        # 处理rewards - 如果是2D则sum到1D
        if rewards.dim() > 1:
            reward_scores = rewards.sum(dim=-1).cpu().tolist()
        else:
            reward_scores = rewards.cpu().tolist()

        # 按uid分组
        uid_to_data = defaultdict(list)
        for i, uid in enumerate(uids):
            uid_to_data[uid].append({
                "response": response_texts[i],
                "reward": reward_scores[i],
            })

        # 构建输出数据
        groups = []
        unique_uids = list(dict.fromkeys(uids))  # 保持顺序的去重

        if max_groups is not None:
            unique_uids = unique_uids[:max_groups]

        for uid in unique_uids:
            # 找到该uid对应的第一个prompt（所有同uid的prompt应该相同）
            first_idx = np.where(uids == uid)[0][0]
            prompt_text = prompt_texts[first_idx]

            groups.append({
                "step": global_step,
                "group_id": str(uid),
                "prompt": prompt_text,
                "responses": uid_to_data[uid],
            })

        # 保存到文件
        output_path = os.path.join(self.log_dir, f"step_{global_step}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(groups, f, ensure_ascii=False, indent=2)

        return output_path


def log_grpo_groups(
    batch,
    tokenizer,
    global_step: int,
    log_dir: str,
    log_freq: int = 10,
    max_groups: Optional[int] = 50,
):
    """
    便捷函数：从batch中提取数据并记录groups

    Args:
        batch: DataProto对象，包含prompts, responses, token_level_scores, uid
        tokenizer: 用于解码的tokenizer
        global_step: 当前训练步数
        log_dir: 日志保存目录
        log_freq: 记录频率
        max_groups: 最大记录的group数量
    """
    if global_step % log_freq != 0:
        return None

    logger = GRPOGroupLogger(log_dir, log_freq)

    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]
    rewards = batch.batch.get("token_level_scores", batch.batch.get("rewards"))
    uids = batch.non_tensor_batch["uid"]

    return logger.log_groups(
        global_step=global_step,
        prompts=prompts,
        responses=responses,
        rewards=rewards,
        uids=uids,
        tokenizer=tokenizer,
        max_groups=max_groups,
    )
