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
python3 scripts/preprocess_pku_saferlhf.py \
    --local_dataset_path Data/PKU-SafeRLHF \
    --local_save_dir Data/pku_saferlhf_verl
```

### 2. 验证（可选）
```bash
# 测试RM加载
python3 scripts/test_rm_loading.py

# 测试数据加载
python3 scripts/test_data_loading.py
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

## Reward/Score 变量说明

本项目的 Reward Model 输出的是**不安全程度**（数值越高越危险），在代码中已做符号反转，转换为**安全程度**（数值越高越安全）。

### 代码中的变量

| 变量 | 位置 | 含义 |
|------|------|------|
| `rm_scores` | `fsdp_workers.py` | `sigmoid(-logits)` 后的**安全分数**（0~1），只在最后一个 response token 位置非零 |
| `token_level_scores` | `ray_trainer.py` | 等于 `rm_scores`，每个token位置的**安全分数** |
| `token_level_rewards` | `ray_trainer.py` | `token_level_scores - KL惩罚`，用于计算advantage的**最终奖励** |

### Wandb 指标

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| `critic/score/mean` | `token_level_scores.sum(-1).mean()` | 序列平均**安全分数**（不含KL惩罚）|
| `critic/score/max` | `token_level_scores.sum(-1).max()` | 序列最高安全分数 |
| `critic/score/min` | `token_level_scores.sum(-1).min()` | 序列最低安全分数 |
| `critic/rewards/mean` | `token_level_rewards.sum(-1).mean()` | 序列平均**最终奖励**（含KL惩罚）|
| `critic/rewards/max` | `token_level_rewards.sum(-1).max()` | 序列最高最终奖励 |
| `critic/rewards/min` | `token_level_rewards.sum(-1).min()` | 序列最低最终奖励 |

### 训练目标

- **RL目标**：最大化 `token_level_rewards`（即最大化安全程度，同时控制与参考模型的KL散度）
- **观察指标**：训练过程中 `critic/score/mean` 应逐步**上升**，表示模型生成的内容越来越安全

### 符号反转位置

```python
# verl/workers/fsdp_workers.py
# RM 输出的是 "不安全程度" logits (越大越不安全)
# 先做 sigmoid(-logits) 得到 (0,1) 的安全分数，再放到最后一个 response token 上
safe_scores = torch.sigmoid(-scores)
token_level_scores = self._expand_to_token_level(data, safe_scores)
output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})
```

## 关键配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `algorithm.adv_estimator` | `grpo` | 使用GRPO算法 |
| `actor_rollout_ref.rollout.n` | `4` | 每个prompt采样4个response |
| `reward_model.model.trust_remote_code` | `True` | 加载自定义RM |
| `trainer.logger` | `["console","wandb"]` | 启用wandb日志 |
