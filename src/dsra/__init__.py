"""DSRA 模块初始化。

提供流式长序列注意力机制的核心实现。

核心组件：
- MultiHeadDSRA2: 多头流式注意力层
- PagedExactMemory: 分页精确记忆引擎
- AttentionLayerSpec: 注意力规格定义
"""

from .domain import *  # noqa: F401,F403
from .mhdsra2 import *  # noqa: F401,F403

__all__ = [
    # 从 domain 导出
    'AttentionLayerSpec',
    'normalize_model_type',
    # 从 mhdsra2 导出
    'MultiHeadDSRA2',
    'MHDSRA2Config',
    'MHDSRA2State',
    'PagedExactMemory',
    'estimate_attention_memory_bytes',
    'format_bytes',
]
