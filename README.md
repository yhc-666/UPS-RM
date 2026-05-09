# UPS-RM: GRPO Training with Custom Reward Model

使用PKU-SafeRLHF数据集和自定义Reward Model进行GRPO强化学习训练。

## 项目 Setting

| 组件 | 配置 |
|------|------|
| Policy Model | Qwen2-7B-Instruct |
| Reward Model | LLaMA3-8B + MLP head |
| 算法 | GRPO (无Critic) |
| 数据集 | PKU-SafeRLHF |
| 环境 | 8×H120 140G GPU |

## Project Tree

```
UPS-RM/
├── scripts/
│   ├── preprocess_pku_saferlhf.py   # 数据预处理
│   ├── run_grpo_pku_saferlhf.sh     # GRPO训练脚本
│   ├── test_rm_loading.py           # RM加载测试
│   └── test_data_loading.py         # 数据加载测试
│
├── merge/
│   └── merged_model/
│       └── Naive-RM-saferlhf/       # 自定义RM模型
│           ├── config.json
│           ├── modeling_myrm.py     # 模型定义
│           └── configuration_myrm.py
│
├── Data/
│   ├── PKU-SafeRLHF/                # 原始数据集
│   └── pku_saferlhf_verl/           # 预处理后数据
│       ├── train.parquet
│       └── val.parquet
│
└── verl/                            # VERL框架
```

## 环境配置

### Wandb登录

```bash
wandb login
```

## 运行步骤

### 1. 数据预处理 (可跳过，已做好)
```bash
python3 scripts/preprocess_pku_saferlhf.py \
    --local_dataset_path Data/PKU-SafeRLHF \
    --local_save_dir Data/pku_saferlhf_verl
```

### 2. Test（不用做）
```bash
python3 scripts/test_rm_loading.py

python3 scripts/test_data_loading.py
```

### 3. 启动训练(记得在里面做路径hyperparam等自定义配置)
```bash
# 修改脚本中的路径后运行
bash scripts/run_grpo_pku_saferlhf.sh
```

## Checkpoint 合并

train完将 FSDP 分片 checkpoint 合并为 HuggingFace 格式：

```bash
python3 -m verl.model_merger merge \
      --backend fsdp \
      --local_dir checkpoints/grpo_pku_saferlhf/Naive-RM-saferlhf-qwen3-4b/global_step_81/actor \
      --target_dir checkpoints/grpo_pku_saferlhf/Naive-RM-saferlhf-qwen3-4b/global_step_81/actor_merged \
      --trust-remote-code

python3 -m verl.model_merger merge \
      --backend fsdp \
      --local_dir checkpoints/grpo_pku_saferlhf/ReCRec-RM-saferlhf-qwen3-4b/global_step_50/actor \
      --target_dir checkpoints/grpo_pku_saferlhf/ReCRec-RM-saferlhf-qwen3-4b/global_step_50/actor_merged \
      --trust-remote-code

python3 -m verl.model_merger merge \
      --backend fsdp \
      --local_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/zhuyanyu/toolfoundry_tensorboard/4b-gspo-user_data_20260127_225111/global_step_60/actor \
      --target_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/zhuyanyu/toolfoundry_tensorboard/4b-gspo-user_data_20260127_225111/global_step_60/actor_merged \
      --trust-remote-code
```