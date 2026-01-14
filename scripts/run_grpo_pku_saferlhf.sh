#!/bin/bash
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

# GRPO Training Script for PKU-SafeRLHF
# Policy: Qwen2-7B-Instruct
# Reward Model: Custom LLaMA3-8B + MLP head
# Environment: 8×H120 140G GPU

set -x

# ========== 路径配置 (请根据实际环境修改) ==========
PROJECT_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/yanghaocheng04/UPS-RM
POLICY_MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen2.5-7B-Instruct
REWARD_MODEL_PATH=${PROJECT_ROOT}/merge/merged_model/Naive-RM-saferlhf

# 数据路径
TRAIN_DATA=${PROJECT_ROOT}/Data/pku_saferlhf_verl/train.parquet
VAL_DATA=${PROJECT_ROOT}/Data/pku_saferlhf_verl/val.parquet

# 输出目录
OUTPUT_DIR=${PROJECT_ROOT}/checkpoints/grpo_pku_saferlhf

# Group logging目录
GROUP_LOG_DIR=${PROJECT_ROOT}/logging

# Group logging采样配置：每次记录x个group，每个group记录y条
GROUP_LOG_MAX_GROUPS=2
GROUP_LOG_MAX_SAMPLES_PER_GROUP=4

# ========== Wandb配置 ==========
export WANDB_PROJECT="verl_grpo_pku_saferlhf"
export WANDB_RUN_NAME="qwen2_7b_instructmyrm_grpo"

# ========== 训练配置 ==========
python3 -X faulthandler -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path=${POLICY_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    reward_model.enable=True \
    reward_model.use_reward_loop=False \
    reward_model.strategy=fsdp \
    reward_model.model.path=${REWARD_MODEL_PATH} \
    reward_model.model.trust_remote_code=True \
    reward_model.model.input_tokenizer=${POLICY_MODEL_PATH} \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    reward_model.micro_batch_size_per_gpu=16 \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_pku_saferlhf' \
    trainer.experiment_name='qwen2_7b_myrm_grpo' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.group_log_dir=${GROUP_LOG_DIR} \
    trainer.group_log_freq=10 \
    trainer.group_log_max_groups=${GROUP_LOG_MAX_GROUPS} \
    trainer.group_log_max_samples_per_group=${GROUP_LOG_MAX_SAMPLES_PER_GROUP} \
    $@
