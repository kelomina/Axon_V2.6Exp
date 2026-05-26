"""Axon v2.6 训练器模块。

提供模型训练、验证和测试的完整流程。
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR, SequentialLR

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from .config import AxonExperimentConfig, TrainingConfig
from .model import AxonMalwareModel


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    phase: str  # train, val, test
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: Optional[float] = None
    learning_rate: float = 0.0
    batch_time: float = 0.0


class EarlyStopping:
    """早停机制"""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max"  # max for metrics like accuracy, min for loss
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """判断是否应该早停"""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.history: List[TrainingMetrics] = []
        self.best_metrics: Dict[str, float] = {}
    
    def update(self, metrics: TrainingMetrics):
        """更新指标"""
        self.history.append(metrics)
    
    def get_best(self, metric_name: str, mode: str = "max") -> Optional[float]:
        """获取最佳指标"""
        if not self.history:
            return None
        
        values = [getattr(m, metric_name) for m in self.history if hasattr(m, metric_name) and getattr(m, metric_name) is not None]
        
        if not values:
            return None
        
        if mode == "max":
            return max(values)
        else:
            return min(values)
    
    def save(self, path: Path):
        """保存历史"""
        data = [
            {
                'epoch': m.epoch,
                'phase': m.phase,
                'loss': m.loss,
                'accuracy': m.accuracy,
                'precision': m.precision,
                'recall': m.recall,
                'f1': m.f1,
                'auc': m.auc,
                'learning_rate': m.learning_rate,
            }
            for m in self.history
        ]
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Path):
        """加载历史"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.history = [
            TrainingMetrics(**d) for d in data
        ]


class AxonTrainer:
    """Axon 模型训练器"""
    
    def __init__(
        self,
        model: AxonMalwareModel,
        config: AxonExperimentConfig,
        train_config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.train_config = train_config or TrainingConfig()
        
        # 设备
        if device is None:
            device = config.get_device()
        self.device = device
        self.model.to(device)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 损失函数
        self.criterion = self._create_criterion()
        
        # 指标追踪
        self.metrics_tracker = MetricsTracker()
        self.best_f1 = 0.0
        self.best_epoch = 0
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            mode="max"
        )
        
        # 输出目录
        self.output_dir = Path(config.model_save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志目录
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self):
        """创建优化器"""
        if self.train_config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
                betas=self.train_config.betas,
                eps=self.train_config.eps
            )
        elif self.train_config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
                momentum=0.9
            )
        else:  # adamw
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
                betas=self.train_config.betas,
                eps=self.train_config.eps
            )
        
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.train_config.lr_scheduler == "none":
            return None
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=self.train_config.warmup_start_lr / self.train_config.learning_rate,
            end_factor=1.0,
            total_iters=self.train_config.warmup_epochs
        )
        
        if self.train_config.lr_scheduler == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_config.max_epochs - self.train_config.warmup_epochs,
                eta_min=self.train_config.min_lr
            )
        elif self.train_config.lr_scheduler == "step":
            main_scheduler = StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            return None
        
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.train_config.warmup_epochs]
        )
        
        return scheduler
    
    def _create_criterion(self):
        """创建损失函数"""
        if self.train_config.focal_gamma > 0:
            # Focal Loss for imbalanced data
            def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
                probs = torch.sigmoid(logits)
                ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                p_t = probs * targets + (1 - probs) * (1 - targets)
                focal_weight = (1 - p_t) ** gamma
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
                return (alpha_t * focal_weight * ce_loss).mean()
            return focal_loss
        else:
            return nn.CrossEntropyLoss(
                label_smoothing=self.train_config.label_smoothing
            )
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> TrainingMetrics:
        """训练一个 epoch"""
        self.model.train()
        
        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (byte_seq, pe_features, stat_features, labels) in enumerate(train_loader):
            # 数据移到设备
            byte_seq = byte_seq.to(self.device)
            pe_features = pe_features.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(byte_seq, pe_features)
            logits = outputs['logits']
            
            # 计算损失
            loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.train_config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.gradient_clip
                )
            
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            num_batches += 1
            
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
        
        batch_time = time.time() - start_time
        
        # 计算指标
        metrics = self._compute_metrics(
            epoch, "train",
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_probs),
            total_loss / num_batches,
            self.optimizer.param_groups[0]['lr'],
            batch_time
        )
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
        epoch: int,
        phase: str = "val"
    ) -> TrainingMetrics:
        """评估模型"""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0.0
        num_batches = 0
        
        for byte_seq, pe_features, stat_features, labels in eval_loader:
            byte_seq = byte_seq.to(self.device)
            pe_features = pe_features.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(byte_seq, pe_features)
            logits = outputs['logits']
            
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1
            
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        metrics = self._compute_metrics(
            epoch, phase,
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_probs),
            total_loss / num_batches,
            self.optimizer.param_groups[0]['lr'],
            0.0
        )
        
        return metrics
    
    def _compute_metrics(
        self,
        epoch: int,
        phase: str,
        targets: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray,
        loss: float,
        lr: float,
        batch_time: float
    ) -> TrainingMetrics:
        """计算评估指标"""
        accuracy = accuracy_score(targets, preds)
        precision = precision_score(targets, preds, zero_division=0)
        recall = recall_score(targets, preds, zero_division=0)
        f1 = f1_score(targets, preds, zero_division=0)
        
        try:
            auc = roc_auc_score(targets, probs)
        except:
            auc = None
        
        metrics = TrainingMetrics(
            epoch=epoch,
            phase=phase,
            loss=loss,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
            learning_rate=lr,
            batch_time=batch_time
        )
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[TrainingMetrics]]:
        """完整训练流程"""
        
        results = {
            'train': [],
            'val': [],
            'test': []
        }
        
        print(f"\n{'='*60}")
        print(f"Starting training: {self.config.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.train_config.max_epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, self.train_config.max_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            results['train'].append(train_metrics)
            self.metrics_tracker.update(train_metrics)
            
            # 打印训练指标
            self._print_metrics(train_metrics, prefix="Train")
            
            # 验证
            if val_loader is not None and epoch % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader, epoch, "val")
                results['val'].append(val_metrics)
                self.metrics_tracker.update(val_metrics)
                self._print_metrics(val_metrics, prefix="Val")
                
                # 保存最佳模型
                if val_metrics.f1 > self.best_f1:
                    self.best_f1 = val_metrics.f1
                    self.best_epoch = epoch
                    self.save_checkpoint("best_model.pt")
                    print(f"  [Best model saved] F1: {val_metrics.f1:.4f}")
                
                # 早停检查
                if self.early_stopping(val_metrics.f1):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
        
        # 测试集评估
        if test_loader is not None:
            print(f"\n{'='*60}")
            print("Evaluating on test set...")
            test_metrics = self.evaluate(test_loader, self.best_epoch, "test")
            results['test'].append(test_metrics)
            self._print_metrics(test_metrics, prefix="Test")
            print(f"{'='*60}\n")
        
        # 保存最终模型
        self.save_checkpoint("final_model.pt")
        
        # 保存训练历史
        self.metrics_tracker.save(self.log_dir / "training_history.json")
        
        return results
    
    def _print_metrics(self, metrics: TrainingMetrics, prefix: str = ""):
        """打印指标"""
        auc_str = f", AUC: {metrics.auc:.4f}" if metrics.auc else ""
        print(
            f"{prefix} | Epoch: {metrics.epoch:3d} | "
            f"Loss: {metrics.loss:.4f} | "
            f"Acc: {metrics.accuracy:.4f} | "
            f"Prec: {metrics.precision:.4f} | "
            f"Rec: {metrics.recall:.4f} | "
            f"F1: {metrics.f1:.4f}{auc_str} | "
            f"LR: {metrics.learning_rate:.2e}"
        )
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'config': self.config.to_dict(),
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_epoch = checkpoint['epoch']
        self.best_f1 = checkpoint['best_f1']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def predict(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """预测单个样本
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        
        with torch.no_grad():
            byte_seq = byte_seq.to(self.device)
            pe_features = pe_features.to(self.device)
            
            logits = self.model(byte_seq, pe_features)['logits']
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        return preds.cpu().numpy(), probs[:, 1].cpu().numpy()


def train_model(
    model: AxonMalwareModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[AxonExperimentConfig] = None,
    train_config: Optional[TrainingConfig] = None,
) -> Tuple[AxonMalwareModel, Dict]:
    """便捷训练函数"""
    
    if config is None:
        config = AxonExperimentConfig()
    if train_config is None:
        train_config = TrainingConfig()
    
    trainer = AxonTrainer(model, config, train_config)
    results = trainer.train(train_loader, val_loader)
    
    return model, results
