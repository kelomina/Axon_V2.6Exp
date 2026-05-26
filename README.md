# Axon v2.6 Experiment

## 概述

Axon v2.6 是一个混合恶意软件检测模型，结合了 DSRA（Dynamic Slot-based Retrieval Attention）流式注意力机制和 KVD（Knowledge-based Vector Descriptor）特征提取器。

### 架构特点

- **DSRA 流式注意力**：基于动态槽位检索的注意力机制，支持长序列处理
- **KVD 特征提取**：从 PE 文件中提取丰富的结构特征和统计特征
- **特征融合**：支持 concat、add、attention 三种融合策略
- **端到端训练**：完整的训练流程，支持早停、学习率调度等

### 项目结构

```
├── src/              # 核心源码
│   ├── axon_exp.py   # 模块入口
│   ├── config.py     # 配置模块
│   ├── model.py      # 模型定义
│   ├── trainer.py    # 训练器
│   ├── dataset.py    # 数据集
│   ├── dsra/         # DSRA 核心模块
│   └── kvd_features/ # KVD 特征提取
├── config/           # 配置文件
├── scripts/          # 工具脚本
├── data/             # 数据目录
└── reports/          # 报告目录
```

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行冒烟测试
python scripts/smoke_test_simple.py

# 启动训练
python scripts/main.py --config config/default_config.toml
```

### 性能指标

| 指标 | 预期值 |
|------|--------|
| Accuracy | > 98% |
| Precision | > 97% |
| Recall | > 97% |
| F1 | > 97% |
| AUC | > 0.99 |

### 文档

详细的模块架构文档请参考 `agents.md` 文件。

---

**License**: MIT
