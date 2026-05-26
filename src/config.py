"""Axon v2.6 实验配置模块。

定义模型、数据集和训练的配置文件。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path


@dataclass
class AxonExperimentConfig:
    """Axon v2.6 实验配置"""
    
    # ==================== 数据配置 ====================
    max_byte_length: int = 65536  # 最大字节序列长度
    pe_feature_dim: int = 1500   # PE 结构特征维度
    stat_feature_dim: int = 100   # 统计特征维度
    batch_size: int = 16          # 批次大小
    
    # ==================== 模型架构配置 ====================
    # 字节嵌入
    byte_embedding_dim: int = 128     # 字节嵌入维度
    use_byte_embedding: bool = True  # 是否使用字节嵌入
    
    # DSRA 配置
    dsra_dim: int = 128               # DSRA 隐藏维度
    dsra_heads: int = 4               # DSRA 注意力头数
    dsra_slots: int = 128             # 记忆槽数量
    dsra_read_topk: int = 8           # 读取 top-k
    dsra_write_topk: int = 4          # 写入 top-k
    dsra_local_window: int = 256      # 局部窗口大小
    dsra_chunk_size: int = 512        # 流式块大小
    
    # PE 特征处理
    pe_projection_dim: int = 128      # PE 特征投影维度
    use_pe_attention: bool = False    # 是否使用 PE 注意力
    
    # 融合配置
    fusion_type: str = "concat"       # 融合类型: concat, add, attention
    dropout: float = 0.1             # Dropout 比率
    
    # ==================== 分类配置 ====================
    num_classes: int = 2              # 分类类别数（二分类：恶意/良性）
    use_class_weights: bool = True    # 是否使用类别权重
    
    # ==================== 训练配置 ====================
    learning_rate: float = 1e-4       # 学习率
    weight_decay: float = 1e-5        # 权重衰减
    max_epochs: int = 50             # 最大训练轮数
    early_stopping_patience: int = 5  # 早停耐心值
    gradient_clip: float = 1.0        # 梯度裁剪阈值
    
    # 优化器
    optimizer: str = "adamw"          # 优化器类型: adam, adamw, sgd
    
    # 学习率调度
    lr_scheduler: str = "cosine"      # 学习率调度: cosine, step, none
    warmup_epochs: int = 3            # 预热轮数
    
    # ==================== 路径配置 ====================
    data_dir: Optional[str] = None    # 数据目录
    model_save_dir: str = "models"    # 模型保存目录
    log_dir: str = "reports/logs"     # 日志目录
    
    # ==================== 实验配置 ====================
    experiment_name: str = "axon_v2.6_exp"
    seed: int = 42                   # 随机种子
    use_wandb: bool = False          # 是否使用 wandb
    device: str = "cuda"             # 设备: cuda, cpu
    
    # ==================== 评估配置 ====================
    eval_interval: int = 1           # 评估间隔（轮）
    save_best_only: bool = True      # 只保存最佳模型
    
    # ==================== 辅助函数 ====================
    def __post_init__(self):
        """配置后处理"""
        # 确保路径为 Path 对象
        if self.model_save_dir:
            self.model_save_dir = Path(self.model_save_dir)
        if self.log_dir:
            self.log_dir = Path(self.log_dir)
        
        # 参数验证
        if self.dsra_dim % self.dsra_heads != 0:
            raise ValueError(f"dsra_dim ({self.dsra_dim}) must be divisible by dsra_heads ({self.dsra_heads})")
        
        if self.fusion_type not in ["concat", "add", "attention"]:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
    
    def get_device(self):
        """获取计算设备"""
        import torch
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    
    def to_dict(self):
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif hasattr(value, '__dataclass_fields__'):
                result[key] = value.to_dict() if hasattr(value, 'to_dict') else str(value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置"""
        # 转换路径字符串为 Path 对象
        if 'model_save_dir' in config_dict:
            config_dict['model_save_dir'] = Path(config_dict['model_save_dir'])
        if 'log_dir' in config_dict:
            config_dict['log_dir'] = Path(config_dict['log_dir'])
        return cls(**config_dict)


@dataclass
class DSRAArchitectureConfig:
    """DSRA 架构特定配置"""
    
    # 基础配置
    dim: int = 128                   # 隐藏维度
    heads: int = 4                   # 注意力头数
    slots: int = 128                 # 全局记忆槽数量
    read_topk: int = 8               # 读取 top-k
    write_topk: int = 4              # 写入 top-k
    local_window: int = 256           # 局部注意力窗口
    
    # 位置编码
    pe_mode: str = "rope"            # 位置编码模式: none, rope, alibi, timestamps
    
    # 可选机制
    use_local: bool = True           # 是否使用局部注意力
    use_retrieval: bool = False      # 是否使用检索注意力
    use_context_film: bool = False   # 是否使用 CCFM 调制
    momentum_qkv: bool = False        # 是否使用 Momentum-QKV
    slot_pe: str = "rope"            # 槽位位置编码: none, rope
    
    # 路由策略
    hard_write: bool = False         # 硬写入路由
    hard_read: bool = False          # 硬读取路由
    exact_write: bool = False       # 精确写入模式
    exact_read: bool = False        # 精确读取模式
    
    # 温度参数
    tau_init: float = 8.0            # 读取温度初始化
    tau_write_init: float = 4.0     # 写入温度初始化
    retrieval_tau: float = 8.0       # 检索温度
    
    # 遗忘机制
    forget_base: float = 0.001       # 基础遗忘率
    forget_conflict: float = 0.20    # 冲突遗忘率
    forget_age: float = 0.0002       # 年龄遗忘率
    
    # 使用 decay
    usage_decay: float = 0.995       # 使用频率衰减
    conf_decay: float = 0.999        # 置信度衰减
    usage_prior: float = 0.25         # 使用频率先验
    
    # 写入策略
    write_frequency: int = 1         # 写入频率
    novelty_threshold: float = 0.0  # 新颖性阈值
    write_protection: int = 0        # 写入保护
    write_gate_min: float = 0.2      # 写入门下限
    
    # 偏差参数
    age_write_bias: float = 0.02    # 年龄写入偏差
    conf_read_bias: float = 0.50     # 置信读取偏差
    age_read_penalty: float = 0.005  # 年龄读取惩罚
    conflict_protection_coef: float = 0.3  # 冲突保护系数


@dataclass
class TrainingConfig:
    """训练特定配置"""
    
    # 优化器
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    
    # 学习率调度
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 3
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6
    
    # 梯度
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    
    # 训练策略
    max_epochs: int = 50
    early_stopping_patience: int = 5
    eval_interval: int = 1
    
    # 批次
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    
    # 损失函数
    label_smoothing: float = 0.0
    focal_gamma: float = 0.0  # 0 表示不使用 focal loss


@dataclass
class DataAugmentationConfig:
    """数据增强配置"""
    
    enable: bool = False              # 是否启用数据增强
    
    # 字节级增强
    byte_dropout: float = 0.0         # 字节 dropout 比率
    byte_swap: float = 0.0            # 字节交换比率
    byte_noise: float = 0.0           # 字节噪声标准差
    
    # 特征级增强
    feature_noise: float = 0.0        # 特征噪声标准差
    feature_mask: float = 0.0          # 特征掩码比率
    
    # Mixup
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    
    # CutMix
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
