# UPS-RM: GRPO Training with Custom Reward Model

基于VERL框架，使用PKU-SafeRLHF数据集和自定义Reward Model进行GRPO强化学习训练。

## 项目配置

| 组件 | 配置 |
|------|------|
| Policy Model | Qwen2-7B-Instruct |
| Reward Model | LLaMA3-8B + MLP head |
| 算法 | GRPO (无Critic) |
| 数据集 | PKU-SafeRLHF |
| 环境 | 8×H120 140G GPU |

## 关键文件结构

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

训练前需要登录wandb以记录实验日志：

```bash
# 方法1: 命令行登录（推荐）
wandb login

# 方法2: 使用API Key环境变量
export WANDB_API_KEY="your-api-key"

# 方法3: 离线模式（不上传到云端）
export WANDB_MODE=offline
```

获取API Key: 访问 https://wandb.ai/authorize

## 运行步骤

### 1. 数据预处理
```bash
python scripts/preprocess_pku_saferlhf.py \
    --local_dataset_path Data/PKU-SafeRLHF \
    --local_save_dir Data/pku_saferlhf_verl
```

### 2. 验证（可选）
```bash
# 测试RM加载
python scripts/test_rm_loading.py

# 测试数据加载
python scripts/test_data_loading.py
```

### 3. 启动训练
```bash
# 修改脚本中的路径后运行
bash scripts/run_grpo_pku_saferlhf.sh
```

### 4. 监控训练
训练日志通过wandb记录，访问 [wandb.ai](https://wandb.ai) 查看：
- `reward/mean` - 平均奖励
- `actor/loss` - Actor损失
- `kl/mean` - KL散度

## 关键配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `algorithm.adv_estimator` | `grpo` | 使用GRPO算法 |
| `actor_rollout_ref.rollout.n` | `4` | 每个prompt采样4个response |
| `reward_model.model.trust_remote_code` | `True` | 加载自定义RM |
| `trainer.logger` | `["console","wandb"]` | 启用wandb日志 |
