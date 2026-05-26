# Axon v2.6 模块架构文档

## 硬约束

在执行任何操作之前，必须通过 `AskUserQuestion` 工具向用户确认，获得明确授权后才能继续。适用场景包括但不限于：

| 操作类型 | 是否需要确认 | 说明 |
|----------|-------------|------|
| 修改核心配置参数 | **是** | 如模型架构、训练超参数等 |
| 修改代码逻辑 | **是** | 涉及模型前向传播、损失计算等核心逻辑 |
| 添加/删除模块 | **是** | 影响项目结构的变更 |
| 执行训练/测试 | **是** | 可能消耗大量计算资源的操作 |
| 提交代码到仓库 | **是** | 所有代码提交前必须确认 |
| 安装依赖包 | **是** | 可能影响环境的操作 |
| 配置环境变量 | **是** | 系统级配置变更 |
| 数据文件修改 | **是** | 涉及数据集的变更 |

### 确认流程

1. **识别操作类型**：判断即将执行的操作是否属于上表中的需要确认的类型
2. **调用 AskUserQuestion**：向用户清晰说明操作内容、影响范围和潜在风险
3. **等待用户反馈**：收到用户确认后才能继续执行
4. **记录确认信息**：保存用户确认记录，便于追溯

### 命令行环境约束

| 约束项 | 要求 | 说明 |
|--------|------|------|
| Shell类型 | PowerShell | 环境默认使用 PowerShell |
| 命令连接符 | `;` | 多个命令串联使用分号，**禁止使用 `&&`** |
| Git路径 | `C:\Program Files\Git\bin\git.exe` | Git不在环境变量中，必须使用绝对路径 |

### Git 操作规范

所有 Git 操作必须使用完整绝对路径：

```powershell
# 正确：使用绝对路径
C:\Program Files\Git\bin\git.exe status
C:\Program Files\Git\bin\git.exe add .
C:\Program Files\Git\bin\git.exe commit -m "message"
C:\Program Files\Git\bin\git.exe push origin main

# 错误：使用相对路径
git status  # ❌ Git不在PATH中
```

### 命令串联示例

```powershell
# 正确：使用分号连接
cd E:\Project\python\Axon_v2.6Exp; C:\Program Files\Git\bin\git.exe status

# 错误：使用 &&
cd E:\Project\python\Axon_v2.6Exp && git status  # ❌ 不支持
```

---

## 概述

Axon v2.6 是一个混合恶意软件检测模型，结合了 DSRA（Dynamic Slot-based Retrieval Attention）流式注意力机制和 KVD（Knowledge-based Vector Descriptor）特征提取器。本文档详细描述项目的核心模块架构。

---

## 项目架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Axon v2.6 Experiment                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌──────────────────┐                      │
│  │   数据层         │    │   配置层         │                      │
│  │  MalwareDataset  │    │  AxonConfig      │                      │
│  │  NPZDataLoader   │    │  DSRAConfig      │                      │
│  └────────┬─────────┘    └────────┬─────────┘                      │
│           │                       │                                 │
│           ▼                       ▼                                 │
│  ┌───────────────────────────────────────────────────────┐         │
│  │                    特征提取层                          │         │
│  │  ┌───────────────┐    ┌───────────────────────────┐   │         │
│  │  │ KVD Extractor │    │    Byte Embedding         │   │         │
│  │  │  PE Features  │    │  + Positional Encoding    │   │         │
│  │  └───────┬───────┘    └───────────┬───────────────┘   │         │
│  │          │                        │                    │         │
│  │          ▼                        ▼                    │         │
│  │  ┌───────────────┐    ┌───────────────────────────┐   │         │
│  │  │ PE Projector  │    │      DSRA Encoder         │   │         │
│  │  └───────┬───────┘    │  (Multi-Head DSRA v2)     │   │         │
│  └──────────┼────────────┴───────────────┬───────────┘   │         │
│             │                            │               │         │
│             └───────────┬────────────────┘               │         │
│                         ▼                                │         │
│              ┌───────────────────┐                        │         │
│              │    Fusion Layer   │                        │         │
│              │  (concat/add/attn)│                        │         │
│              └──────────┬────────┘                        │         │
│                         ▼                                │         │
│              ┌───────────────────┐                        │         │
│              │   Classifier Head │                        │         │
│              └──────────┬────────┘                        │         │
│                         ▼                                │         │
│              ┌───────────────────┐                        │         │
│              │    Trainer        │                        │         │
│              └───────────────────┘                        │         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心模块

### 1. 配置模块 (`src/config.py`)

提供实验配置和DSRA架构配置。

#### 1.1 AxonExperimentConfig

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `max_byte_length` | int | 65536 | 最大字节序列长度 |
| `pe_feature_dim` | int | 1500 | PE结构特征维度 |
| `byte_embedding_dim` | int | 128 | 字节嵌入维度 |
| `dsra_dim` | int | 128 | DSRA隐藏维度 |
| `dsra_heads` | int | 4 | DSRA注意力头数 |
| `dsra_slots` | int | 128 | 全局记忆槽数量 |
| `fusion_type` | str | "concat" | 融合类型: concat/add/attention |
| `dropout` | float | 0.1 | Dropout比率 |
| `num_classes` | int | 2 | 分类类别数 |

#### 1.2 DSRAArchitectureConfig

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `dim` | int | 128 | 隐藏维度 |
| `heads` | int | 4 | 注意力头数 |
| `slots` | int | 128 | 全局记忆槽数量 |
| `read_topk` | int | 8 | 读取top-k |
| `write_topk` | int | 4 | 写入top-k |
| `local_window` | int | 256 | 局部注意力窗口 |
| `tau_init` | float | 8.0 | 读取温度初始化 |
| `forget_base` | float | 0.001 | 基础遗忘率 |
| `use_local` | bool | True | 是否使用局部注意力 |
| `use_retrieval` | bool | False | 是否使用检索注意力 |

---

### 2. 模型模块 (`src/model.py`)

#### 2.1 组件层次

| 组件 | 职责 | 位置 |
|------|------|------|
| `ByteEmbedding` | 字节值(0-255)到向量的映射 | 第68-89行 |
| `PositionalEncoding` | 位置编码(可学习/RoPE/正弦) | 第16-65行 |
| `PEFeatureProjector` | PE特征投影到低维空间 | 第98-136行 |
| `DSRAEncoder` | DSRA序列编码核心 | 第139-213行 |
| `MalwareDSRAEncoder` | 恶意软件专用DSRA编码器 | 第216-336行 |
| `AxonMalwareModel` | 完整融合模型 | 第339-468行 |

#### 2.2 AxonMalwareModel 架构

```
输入层:
  ├─ byte_seq: [B, max_byte_length]     字节序列
  └─ pe_features: [B, pe_feature_dim]   PE结构特征

特征提取层:
  ├─ ByteEmbedding → PositionalEncoding → DSRAEncoder → byte_repr [B, dsra_dim]
  └─ PEFeatureProjector → pe_repr [B, pe_projection_dim]

融合层 (fusion_type):
  ├─ concat: byte_repr ⊕ pe_repr → [B, dsra_dim + pe_projection_dim]
  ├─ add:    Proj(byte_repr) + pe_repr → [B, pe_projection_dim]
  └─ attention: MHA(byte_repr, pe_repr) → [B, pe_projection_dim]

分类层:
  ├─ LayerNorm → Dropout → Linear(64) → GELU → Dropout → Linear(num_classes)
  └─ Output: logits [B, num_classes]
```

#### 2.3 前向传播流程

```python
def forward(byte_seq, pe_features):
    # 1. 字节序列编码
    byte_emb = self.byte_embedding(byte_seq)           # [B, L, byte_dim]
    byte_emb = self.pos_encoding(byte_emb)             # 添加位置编码
    byte_emb = self.input_proj(byte_emb)               # [B, L, dsra_dim]
    byte_out, state = self.dsra_encoder(byte_emb)      # [B, L, dsra_dim]
    byte_repr = byte_out[:, -1, :]                     # [B, dsra_dim]
    
    # 2. PE特征投影
    pe_repr = self.pe_projector(pe_features)           # [B, pe_proj_dim]
    
    # 3. 特征融合
    if fusion_type == "concat":
        fused = torch.cat([byte_repr, pe_repr], dim=-1)
    elif fusion_type == "add":
        fused = self.fusion(byte_repr) + pe_repr
    else:  # attention
        fused = self.fusion(byte_repr, pe_repr, pe_repr)
    
    # 4. 分类
    logits = self.classifier(fused)                    # [B, num_classes]
    return {'logits': logits}
```

---

### 3. DSRA 核心模块 (`src/dsra/`)

#### 3.1 模块结构

```
src/dsra/
├── __init__.py
├── dsra_layer.py          # DSRA基础层
├── dsra_model.py          # DSRA完整模型
├── mhdsra2/               # 多头DSRA v2实现
│   ├── __init__.py
│   ├── improved_dsra_mha.py    # 核心多头DSRA实现
│   └── paged_exact_memory.py   # 分页精确记忆
└── domain/                # 领域模型
    ├── __init__.py
    ├── model_spec.py      # 模型规格
    ├── attention_spec.py  # 注意力规格
    └── arithmetic_emergence.py # 算术涌现
```

#### 3.2 MultiHeadDSRA2 核心特性

| 特性 | 描述 |
|------|------|
| **动态记忆槽** | 可动态分配和遗忘的全局记忆槽 |
| **局部注意力** | 滑动窗口局部注意力机制 |
| **检索注意力** | 基于内容的记忆检索 |
| **遗忘机制** | 基于年龄、冲突、使用频率的遗忘 |
| **温度调节** | 可学习的读取/写入温度 |
| **写入门控** | 新颖性检测和写保护 |

---

### 4. KVD 特征提取模块 (`src/kvd_features/`)

#### 4.1 Extractor 组件

负责从PE文件中提取结构特征。

| 特征类型 | 维度 | 描述 |
|----------|------|------|
| **导入表** | 512 | 导入函数名称哈希 |
| **节信息** | 128 | 节名称、大小、权限 |
| **头信息** | 64 | PE头元数据 |
| **导出表** | 64 | 导出函数信息 |
| **资源表** | 128 | 资源节特征 |
| **字符串特征** | 256 | 字符串熵和频率 |
| **统计特征** | 100 | 字节级统计 |
| **其他** | 352 | 杂项特征 |
| **总计** | 1500 | PE特征总维度 |

---

### 5. 数据集模块 (`src/dataset.py`)

#### 5.1 MalwareDataset

| 方法 | 功能 |
|------|------|
| `__init__` | 初始化数据集，加载NPZ文件 |
| `__len__` | 返回样本数量 |
| `__getitem__` | 获取单个样本 |
| `collate_fn` | 批次处理函数 |

#### 5.2 数据格式

```python
# 样本结构
{
    'byte_seq': np.ndarray,      # [max_byte_length] uint8
    'pe_features': np.ndarray,   # [pe_feature_dim] float32
    'stat_features': np.ndarray, # [stat_feature_dim] float32
    'label': int,                # 0=良性, 1=恶意
}
```

---

### 6. 训练器模块 (`src/trainer.py`)

#### 6.1 AxonTrainer 功能

| 方法 | 功能 |
|------|------|
| `__init__` | 初始化训练器，创建优化器/调度器/损失函数 |
| `train_epoch` | 训练单个epoch |
| `evaluate` | 评估模型性能 |
| `train` | 完整训练流程 |
| `save_checkpoint` | 保存模型检查点 |
| `load_checkpoint` | 加载模型检查点 |
| `predict` | 预测单个样本 |

#### 6.2 支持的优化器

| 优化器 | 配置参数 |
|--------|----------|
| Adam | `optimizer="adam"` |
| AdamW | `optimizer="adamw"` (默认) |
| SGD | `optimizer="sgd"` |

#### 6.3 学习率调度器

| 调度器 | 配置参数 |
|--------|----------|
| Cosine Annealing | `lr_scheduler="cosine"` (默认) |
| Step LR | `lr_scheduler="step"` |
| None | `lr_scheduler="none"` |

#### 6.4 评估指标

| 指标 | 计算方式 |
|------|----------|
| Accuracy | `sklearn.metrics.accuracy_score` |
| Precision | `sklearn.metrics.precision_score` |
| Recall | `sklearn.metrics.recall_score` |
| F1 | `sklearn.metrics.f1_score` |
| AUC | `sklearn.metrics.roc_auc_score` |

---

## 工具脚本

### scripts/main.py

主训练脚本，支持命令行参数。

```bash
python scripts/main.py \
    --config config/default_config.toml \
    --data_dir /path/to/data \
    --epochs 50 \
    --batch_size 16
```

### scripts/smoke_test.py

冒烟测试脚本，验证模型基本功能。

```bash
python scripts/smoke_test.py
```

---

## 数据流向图

```
原始PE文件
     │
     ▼
┌─────────────────┐
│  KVD Extractor  │  ← 特征提取
└────────┬────────┘
         │ pe_features [1500]
         ▼
┌──────────────────────────────────────────────┐
│           AxonMalwareModel                   │
│  ┌───────────────────┐  ┌─────────────────┐ │
│  │ ByteEmbedding     │  │ PEProjector     │ │
│  │ + PositionalEnc   │  │                 │ │
│  └─────────┬─────────┘  └────────┬────────┘ │
│            │                     │          │
│            ▼                     ▼          │
│  ┌───────────────────┐  ┌─────────────────┐ │
│  │   DSRA Encoder    │  │   pe_repr       │ │
│  │   (流式处理)      │  │   [128]         │ │
│  └─────────┬─────────┘  └────────┬────────┘ │
│            │ byte_repr [128]     │          │
│            └───────────┬─────────┘          │
│                        ▼                    │
│              ┌─────────────────┐            │
│              │    Fusion       │            │
│              │   (concat)      │            │
│              └────────┬────────┘            │
│                       ▼                    │
│              ┌─────────────────┐            │
│              │   Classifier    │            │
│              └────────┬────────┘            │
│                       ▼                    │
│              logits [2] → softmax → [prob]  │
└──────────────────────────────────────────────┘
     │
     ▼
输出: 恶意软件概率 [0.0, 1.0]
```

---

## 配置示例

### 默认配置 (`config/default_config.toml`)

```toml
[experiment]
name = "axon_v2.6_exp"
seed = 42
device = "cuda"

[data]
max_byte_length = 65536
pe_feature_dim = 1500
batch_size = 16

[model]
byte_embedding_dim = 128
dsra_dim = 128
dsra_heads = 4
dsra_slots = 128
fusion_type = "concat"
dropout = 0.1

[training]
learning_rate = 0.0001
weight_decay = 0.00001
max_epochs = 50
early_stopping_patience = 5
optimizer = "adamw"
lr_scheduler = "cosine"
```

---

## 性能指标

### 预期基准

| 指标 | 预期值 |
|------|--------|
| Accuracy | > 98% |
| Precision | > 97% |
| Recall | > 97% |
| F1 | > 97% |
| AUC | > 0.99 |

---

## 扩展指南

### 添加新的融合策略

1. 在 `AxonMalwareModel.__init__` 中添加新的融合类型判断
2. 实现对应的融合层
3. 更新 `fusion_type` 配置验证

### 添加新的特征类型

1. 在 `kvd_features/extractor.py` 中添加特征提取逻辑
2. 更新配置中的 `pe_feature_dim`
3. 更新 `PEFeatureProjector` 输入维度

### 调整 DSRA 参数

修改 `DSRAArchitectureConfig` 中的参数，主要关注：
- `slots`: 记忆槽数量
- `read_topk`/`write_topk`: 读写top-k
- `tau_init`: 温度参数
- `forget_*`: 遗忘率参数

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v2.6.0 | 2024-Q4 | 初始版本，DSRA v2 + KVD特征融合 |
