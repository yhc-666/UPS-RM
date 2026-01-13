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
        max_samples_per_group: Optional[int] = None,
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
            max_samples_per_group: 每个group最大记录的样本数（None表示全部记录）
        """
        # 处理rewards - 如果是2D则sum到1D
        if rewards.dim() > 1:
            reward_scores = rewards.sum(dim=-1)
        else:
            reward_scores = rewards

        reward_scores = reward_scores.detach().cpu().tolist()

        # 选择要记录的groups与indices（单次遍历，保持uids顺序）
        selected_uids = []
        selected_uid_set = set()
        uid_to_indices: dict[object, list[int]] = defaultdict(list)

        for i, uid in enumerate(uids):
            if uid not in selected_uid_set:
                if max_groups is not None and len(selected_uids) >= max_groups:
                    continue
                selected_uids.append(uid)
                selected_uid_set.add(uid)

            if uid in selected_uid_set:
                if max_samples_per_group is None or len(uid_to_indices[uid]) < max_samples_per_group:
                    uid_to_indices[uid].append(i)

        # 过滤掉没有任何样本被选中的group（例如 max_samples_per_group=0）
        selected_uids = [uid for uid in selected_uids if uid_to_indices[uid]]

        if not selected_uids:
            return None

        # 解码所需的prompts/responses（只对采样到的indices解码，避免全batch decode）
        prompt_indices = [uid_to_indices[uid][0] for uid in selected_uids if uid_to_indices[uid]]
        response_indices = [idx for uid in selected_uids for idx in uid_to_indices[uid]]

        prompt_texts = tokenizer.batch_decode(prompts[prompt_indices].cpu(), skip_special_tokens=True)
        response_texts = tokenizer.batch_decode(responses[response_indices].cpu(), skip_special_tokens=True)

        uid_to_prompt_text = {uid: prompt_texts[i] for i, uid in enumerate(selected_uids)}
        idx_to_response_text = {idx: response_texts[i] for i, idx in enumerate(response_indices)}

        # 构建输出数据
        groups = []
        for uid in selected_uids:
            indices = uid_to_indices[uid]
            if not indices:
                continue
            groups.append(
                {
                    "step": global_step,
                    "group_id": str(uid),
                    "prompt": uid_to_prompt_text[uid],
                    "responses": [
                        {
                            "response": idx_to_response_text[idx],
                            "reward": float(reward_scores[idx]),
                        }
                        for idx in indices
                    ],
                }
            )

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
    max_samples_per_group: Optional[int] = None,
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
        max_samples_per_group: 每个group最大记录的样本数
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
        max_samples_per_group=max_samples_per_group,
    )
