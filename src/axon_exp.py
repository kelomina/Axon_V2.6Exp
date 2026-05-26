"""Axon v2.6 实验模块。

本模块提供恶意软件检测的混合模型，结合了：
- KVD 特征提取器（PE 结构特征、统计特征）
- DSRA 流式注意力机制（字节序列处理）

融合架构：
    字节序列 -> DSRA 流式编码 -> 序列特征
        ↓
    融合层 <- PE 特征投影
        ↓
    分类头 -> 恶意软件分类

使用方法：
    from axon_exp import AxonMalwareModel, AxonExperimentConfig
    
    model = AxonMalwareModel(config)
    prediction = model(byte_seq, pe_features)
"""

from .config import AxonExperimentConfig
from .model import AxonMalwareModel, MalwareDSRAEncoder
from .dataset import MalwareDataset, NPZDataLoader
from .trainer import AxonTrainer

__all__ = [
    'AxonExperimentConfig',
    'AxonMalwareModel',
    'MalwareDSRAEncoder',
    'MalwareDataset',
    'NPZDataLoader',
    'AxonTrainer',
]
