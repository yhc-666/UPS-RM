- 任务：使用PKU-Alignment/PKU-SafeRLHF数据集+一个自定义的奖励模型（llm+MLPhead）做单轮 QA + 判别式 RM。我当前正在本地进行verl框架的开发，具体运行会在remote server上。请协助我进行本地开发。

- 环境 8*A100 96G GPU

- Reward model
    - backbone：sfairXC/FsfairX-LLaMA3-RM-v0.1 (8B)
    - 提取sfairXC/FsfairX-LLaMA3-RM-v0.1的last token hiddenstate 然后顶上接了个MLP head 输出logits
    - MLP head是我以前单独训练的
    - 整个Reward model权重通过@merge 合并获得
    - merge后的模型都位于@merge/merged_model，其中的LLM权重在remote server上没有在本地列出。
    - Demo: 在远程server上@merge/merged_model/Naive-RM-saferlhf的具体结构：
        Naive-RM-saferlfh
        ├── .gitattributes
        ├── config.json
        ├── configuration_myrm.py
        ├── model-00001-of-00004.safetensors
        ├── model-00002-of-00004.safetensors
        ├── model-00003-of-00004.safetensors
        ├── model-00004-of-00004.safetensors
        ├── model.safetensors.index.json
        ├── modeling_myrm.py
        ├── myrm.safetensors
        ├── README.md
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        ├── tokenizer.json


- 我想要使用VERL的
    - Legacy FSDP RM
    - AgentLoop