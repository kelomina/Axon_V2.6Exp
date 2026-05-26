"""Axon v2.6 融合模型模块。

结合 KVD 特征提取和 DSRA 流式注意力的恶意软件检测模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .config import AxonExperimentConfig, DSRAArchitectureConfig
from .dsra.mhdsra2 import MultiHeadDSRA2, MHDSRA2Config


class PositionalEncoding(nn.Module):
    """位置编码模块
    
    支持：
    - 可学习的位置编码
    - RoPE (Rotary Position Embedding)
    - 绝对位置编码
    """
    
    def __init__(self, d_model: int, max_len: int = 65536, mode: str = "learnable"):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        
        if mode == "learnable":
            self.pos_embedding = nn.Embedding(max_len, d_model)
        elif mode == "sinusoidal":
            self.register_buffer('pe', self._create_sinusoidal_encoding(max_len, d_model))
        else:
            self.pe = None
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """创建正弦位置编码"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            [B, seq_len, d_model]
        """
        seq_len = x.shape[1]
        
        if self.mode == "learnable":
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            return x + self.pos_embedding(positions)
        elif self.mode == "sinusoidal":
            return x + self.pe[:seq_len].unsqueeze(0)
        else:
            return x


class ByteEmbedding(nn.Module):
    """字节嵌入层
    
    将字节值 (0-255) 映射到嵌入向量。
    """
    
    def __init__(self, vocab_size: int = 256, embedding_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len] 字节值 (0-255)
        Returns:
            [B, seq_len, embedding_dim]
        """
        # Clamp 确保在有效范围内
        x = torch.clamp(x, 0, self.vocab_size - 1)
        return self.embedding(x)


class PEFeatureProjector(nn.Module):
    """PE 特征投影器
    
    将高维 PE 结构特征投影到低维空间。
    """
    
    def __init__(
        self,
        input_dim: int = 1500,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.projector = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] PE 特征
        Returns:
            [B, output_dim]
        """
        return self.projector(x)


class DSRAEncoder(nn.Module):
    """DSRA 序列编码器
    
    使用 MHDSRA2 流式注意力处理字节序列。
    """
    
    def __init__(self, config: DSRAArchitectureConfig):
        super().__init__()
        
        # 创建 MHDSRA2 配置
        mhdsra_config = MHDSRA2Config(
            dim=config.dim,
            heads=config.heads,
            slots=config.slots,
            read_topk=config.read_topk,
            write_topk=config.write_topk,
            local_window=config.local_window,
            use_local=config.use_local,
            use_retrieval=config.use_retrieval,
            tau_init=config.tau_init,
            tau_write_init=config.tau_write_init,
            retrieval_tau=config.retrieval_tau,
            forget_base=config.forget_base,
            forget_conflict=config.forget_conflict,
            forget_age=config.forget_age,
            usage_decay=config.usage_decay,
            conf_decay=config.conf_decay,
            usage_prior=config.usage_prior,
            age_write_bias=config.age_write_bias,
            conf_read_bias=config.conf_read_bias,
            age_read_penalty=config.age_read_penalty,
            use_context_film=config.use_context_film,
            momentum_qkv=config.momentum_qkv,
            slot_pe=config.slot_pe,
            hard_write=config.hard_write,
            hard_read=config.hard_read,
            exact_write=config.exact_write,
            exact_read=config.exact_read,
            write_frequency=config.write_frequency,
            novelty_threshold=config.novelty_threshold,
            write_protection=config.write_protection,
            write_gate_min=config.write_gate_min,
            conflict_protection_coef=config.conflict_protection_coef,
        )
        
        self.dsra = MultiHeadDSRA2(mhdsra_config)
        self.config = config
    
    def forward(
        self,
        x: torch.Tensor,
        state=None,
        return_aux: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict], Optional[any]]:
        """
        Args:
            x: [B, seq_len, dim] 输入序列
            state: 可选的 DSRA 状态
            return_aux: 是否返回辅助信息
        Returns:
            Tuple of (output, next_state, aux)
        """
        if state is None:
            state = self.dsra.init_state(x.shape[0], device=x.device, dtype=x.dtype)
        
        if return_aux:
            out, next_state, aux = self.dsra(x, state, return_aux=True)
            return out, next_state, aux
        else:
            out, next_state = self.dsra(x, state)
            return out, next_state, None
    
    def init_state(self, batch_size: int, device=None, dtype=None):
        """初始化 DSRA 状态"""
        return self.dsra.init_state(batch_size, device=device, dtype=dtype)


class MalwareDSRAEncoder(nn.Module):
    """恶意软件 DSRA 编码器
    
    结合字节嵌入、位置编码和 DSRA 流式注意力的编码器。
    """
    
    def __init__(
        self,
        byte_embedding_dim: int = 128,
        max_byte_length: int = 65536,
        dsra_config: Optional[DSRAArchitectureConfig] = None,
        pe_feature_dim: int = 1500,
        pe_projection_dim: int = 128,
        use_pos_encoding: bool = True,
        pos_encoding_mode: str = "learnable",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.byte_embedding_dim = byte_embedding_dim
        self.max_byte_length = max_byte_length
        
        # 字节嵌入
        self.byte_embedding = ByteEmbedding(
            vocab_size=256,
            embedding_dim=byte_embedding_dim
        )
        
        # 位置编码
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(
                d_model=byte_embedding_dim,
                max_len=max_byte_length,
                mode=pos_encoding_mode
            )
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(byte_embedding_dim, dsra_config.dim if dsra_config else byte_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # DSRA 配置
        if dsra_config is None:
            dsra_config = DSRAArchitectureConfig(dim=byte_embedding_dim)
        
        # DSRA 编码器
        self.dsra_encoder = DSRAEncoder(dsra_config)
        
        # PE 特征投影
        self.pe_projector = PEFeatureProjector(
            input_dim=pe_feature_dim,
            hidden_dim=pe_projection_dim * 2,
            output_dim=pe_projection_dim,
            dropout=dropout
        )
        
        # 输出维度
        self.output_dim = dsra_config.dim + pe_projection_dim
    
    def forward(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor,
        state=None,
        return_aux: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[any]]:
        """
        Args:
            byte_seq: [B, max_byte_length] 字节序列
            pe_features: [B, pe_feature_dim] PE 结构特征
            state: 可选的 DSRA 状态
            return_aux: 是否返回辅助信息
        Returns:
            Tuple of (byte_repr, pe_repr, state)
            - byte_repr: [B, dsra_dim] 字节序列表示
            - pe_repr: [B, pe_projection_dim] PE 特征表示
            - state: DSRA 状态
        """
        # 字节嵌入
        byte_emb = self.byte_embedding(byte_seq)  # [B, L, byte_emb_dim]
        
        # 位置编码
        if self.use_pos_encoding:
            byte_emb = self.pos_encoding(byte_emb)
        
        # 输入投影
        byte_emb = self.input_proj(byte_emb)  # [B, L, dsra_dim]
        
        # 分块处理（如果需要）
        chunk_size = 512
        seq_len = byte_emb.shape[1]
        
        if seq_len <= chunk_size:
            # 直接处理整个序列
            byte_out, next_state, aux = self.dsra_encoder(byte_emb, state, return_aux=return_aux)
        else:
            # 分块处理长序列
            byte_outs = []
            aux_all = []
            
            for i in range(0, seq_len, chunk_size):
                chunk = byte_emb[:, i:i+chunk_size, :]
                chunk_out, state, chunk_aux = self.dsra_encoder(chunk, state, return_aux=return_aux)
                byte_outs.append(chunk_out)
                if chunk_aux:
                    aux_all.append(chunk_aux)
            
            # 合并块输出（取最后一块的输出）
            byte_out = byte_outs[-1]
            aux = aux_all if aux_all else None
        
        # 序列表示：取最后位置的输出
        byte_repr = byte_out[:, -1, :]  # [B, dsra_dim]
        
        # PE 特征投影
        pe_repr = self.pe_projector(pe_features)  # [B, pe_projection_dim]
        
        return byte_repr, pe_repr, next_state


class AxonMalwareModel(nn.Module):
    """Axon 恶意软件检测模型
    
    融合 DSRA 字节序列编码和 PE 结构特征。
    
    架构：
        字节序列 -> ByteEmbedding -> DSRA 流式编码 -> 序列特征
            ↓
        融合层 <- PE 特征投影
            ↓
        分类头 -> 二分类输出
    """
    
    def __init__(self, config: AxonExperimentConfig):
        super().__init__()
        self.config = config
        
        # DSRA 编码器
        dsra_config = DSRAArchitectureConfig(
            dim=config.dsra_dim,
            heads=config.dsra_heads,
            slots=config.dsra_slots,
            read_topk=config.dsra_read_topk,
            write_topk=config.dsra_write_topk,
            local_window=config.dsra_local_window,
        )
        
        self.dsra_encoder = MalwareDSRAEncoder(
            byte_embedding_dim=config.byte_embedding_dim,
            max_byte_length=config.max_byte_length,
            dsra_config=dsra_config,
            pe_feature_dim=config.pe_feature_dim,
            pe_projection_dim=config.pe_projection_dim,
            dropout=config.dropout,
        )
        
        # 融合层
        fusion_dim = config.dsra_dim + config.pe_projection_dim
        
        if config.fusion_type == "concat":
            self.fusion = nn.Identity()
            classifier_input_dim = fusion_dim
        elif config.fusion_type == "add":
            self.fusion = nn.Linear(config.dsra_dim, config.pe_projection_dim)
            classifier_input_dim = config.pe_projection_dim
        else:  # attention
            self.fusion = nn.MultiheadAttention(
                embed_dim=config.pe_projection_dim,
                num_heads=4,
                dropout=config.dropout
            )
            classifier_input_dim = config.pe_projection_dim
        
        self.fusion_type = config.fusion_type
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(config.dropout),
            nn.Linear(classifier_input_dim, 64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_classes)
        )
        
        # 辅助任务头（可选）
        self.aux_head = None
        if hasattr(config, 'use_aux_task') and config.use_aux_task:
            self.aux_head = nn.Linear(classifier_input_dim, 1)
    
    def forward(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            byte_seq: [B, max_byte_length] 字节序列
            pe_features: [B, pe_feature_dim] PE 结构特征
            return_features: 是否返回中间特征
        Returns:
            Dict containing:
            - logits: [B, num_classes] 分类 logits
            - features: (可选) 融合特征
            - byte_repr: (可选) 字节序列表示
            - pe_repr: (可选) PE 特征表示
        """
        # DSRA 编码
        byte_repr, pe_repr, _ = self.dsra_encoder(byte_seq, pe_features)
        
        # 特征融合
        if self.fusion_type == "concat":
            fused_features = torch.cat([byte_repr, pe_repr], dim=-1)
        elif self.fusion_type == "add":
            # 将 byte_repr 投影到 pe_repr 的维度，然后相加
            projected_byte = self.fusion(byte_repr)
            fused_features = projected_byte + pe_repr
        else:  # attention
            byte_repr_expanded = byte_repr.unsqueeze(0)  # [1, B, dim]
            pe_repr_expanded = pe_repr.unsqueeze(0)  # [1, B, dim]
            attn_out, _ = self.fusion(byte_repr_expanded, pe_repr_expanded, pe_repr_expanded)
            fused_features = attn_out.squeeze(0)
        
        # 分类
        logits = self.classifier(fused_features)
        
        if return_features:
            return {
                'logits': logits,
                'features': fused_features,
                'byte_repr': byte_repr,
                'pe_repr': pe_repr
            }
        
        return {'logits': logits}
    
    def get_logits(self, byte_seq: torch.Tensor, pe_features: torch.Tensor) -> torch.Tensor:
        """获取分类 logits"""
        return self.forward(byte_seq, pe_features)['logits']
    
    def predict_proba(self, byte_seq: torch.Tensor, pe_features: torch.Tensor) -> torch.Tensor:
        """预测类别概率"""
        logits = self.get_logits(byte_seq, pe_features)
        return F.softmax(logits, dim=-1)
    
    def predict(self, byte_seq: torch.Tensor, pe_features: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        proba = self.predict_proba(byte_seq, pe_features)
        return torch.argmax(proba, dim=-1)


class HybridLightGBMModel(nn.Module):
    """混合 LightGBM + DSRA 模型
    
    将预训练的 LightGBM 特征与 DSRA 特征融合。
    """
    
    def __init__(
        self,
        lgb_feature_dim: int = 1500,
        dsra_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # LightGBM 特征处理
        self.lgb_proj = nn.Sequential(
            nn.Linear(lgb_feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        
        # DSRA 编码器
        self.dsra_encoder = MalwareDSRAEncoder(
            byte_embedding_dim=dsra_dim,
            dsra_config=DSRAArchitectureConfig(dim=dsra_dim),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),  # lgb_proj(128) + dsra_dim
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 分类头
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(
        self,
        lgb_features: torch.Tensor,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            lgb_features: [B, lgb_feature_dim] LightGBM 特征
            byte_seq: [B, max_byte_length] 字节序列
            pe_features: [B, pe_feature_dim] PE 特征
        """
        # LightGBM 特征处理
        lgb_repr = self.lgb_proj(lgb_features)
        
        # DSRA 编码
        byte_repr, pe_repr, _ = self.dsra_encoder(byte_seq, pe_features)
        
        # 融合
        fused = torch.cat([lgb_repr, byte_repr], dim=-1)
        fused = self.fusion(fused)
        
        return self.classifier(fused)
