"""Axon v2.6 数据集和加载器模块。

提供恶意软件检测的数据集类和批量加载器。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
from pathlib import Path

from ..kvd_features.extractor import (
    extract_all_features,
    ExtractionConfig
)


class MalwareDataset(Dataset):
    """恶意软件数据集
    
    支持两种数据格式：
    1. 原始文件目录（需要实时提取特征）
    2. NPZ 文件目录（预提取特征）
    
    数据格式：
        byte_sequence: [max_byte_length] uint8
        pe_features: [pe_feature_dim] float32
        stat_features: [stat_feature_dim] float32
        label: int (0=良性, 1=恶意)
    """
    
    def __init__(
        self,
        data_dir: str,
        max_byte_length: int = 65536,
        pe_feature_dim: int = 1500,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        transform=None,
        target_transform=None,
        label_inference: str = "directory"  # directory, filename, metadata
    ):
        """
        Args:
            data_dir: 数据目录路径
            max_byte_length: 最大字节序列长度
            pe_feature_dim: PE 特征维度
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
            transform: 数据转换
            target_transform: 标签转换
            label_inference: 标签推断方式
        """
        self.data_dir = Path(data_dir)
        self.max_byte_length = max_byte_length
        self.pe_feature_dim = pe_feature_dim
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / ".cache"
        self.transform = transform
        self.target_transform = target_transform
        self.label_inference = label_inference
        
        self.extraction_config = ExtractionConfig(
            max_file_size=max_byte_length,
            pe_feature_dim=pe_feature_dim
        )
        
        # 扫描数据文件
        self.file_list = []
        self.label_list = []
        self._scan_directory()
        
        # 创建缓存目录
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _scan_directory(self):
        """扫描数据目录，收集所有文件"""
        benign_dir = self.data_dir / "benign"
        malicious_dir = self.data_dir / "malicious"
        
        # 扫描良性样本
        if benign_dir.exists():
            for file_path in benign_dir.rglob("*"):
                if file_path.is_file() and self._is_valid_sample(file_path):
                    self.file_list.append(file_path)
                    self.label_list.append(0)
        
        # 扫描恶意样本
        if malicious_dir.exists():
            for file_path in malicious_dir.rglob("*"):
                if file_path.is_file() and self._is_valid_sample(file_path):
                    self.file_list.append(file_path)
                    self.label_list.append(1)
        
        # 如果没有子目录，直接扫描根目录
        if not benign_dir.exists() and not malicious_dir.exists():
            for file_path in self.data_dir.rglob("*"):
                if file_path.is_file() and self._is_valid_sample(file_path):
                    self.file_list.append(file_path)
                    self.label_list.append(self._infer_label(file_path))
    
    def _is_valid_sample(self, file_path: Path) -> bool:
        """检查是否为有效的样本文件"""
        valid_extensions = {'.exe', '.dll', '.sys', '.ocx', '.scr', '.bat', '.cmd', '.msi'}
        return file_path.suffix.lower() in valid_extensions or file_path.stat().st_size < 100 * 1024 * 1024
    
    def _infer_label(self, file_path: Path) -> int:
        """从文件名推断标签"""
        filename_lower = file_path.name.lower()
        
        if self.label_inference == "filename":
            # 从文件名推断
            malicious_keywords = ['malware', 'virus', 'trojan', 'backdoor', 'worm', 
                               'ransomware', 'packed', 'packed_', 'malicious']
            benign_keywords = ['benign', 'normal', 'clean', 'legitimate']
            
            for kw in malicious_keywords:
                if kw in filename_lower:
                    return 1
            for kw in benign_keywords:
                if kw in filename_lower:
                    return 0
        elif self.label_inference == "directory":
            # 从父目录推断
            parent_name = file_path.parent.name.lower()
            if "malicious" in parent_name or "malware" in parent_name:
                return 1
            elif "benign" in parent_name:
                return 0
        
        return 0  # 默认良性
    
    def _get_cache_path(self, file_path: Path) -> Path:
        """获取缓存文件路径"""
        file_hash = str(hash(str(file_path)))
        return self.cache_dir / f"{file_hash}.npz"
    
    def _load_from_cache(self, file_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """从缓存加载数据"""
        cache_path = self._get_cache_path(file_path)
        
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                return (
                    data['byte_sequence'],
                    data['pe_features'],
                    data.get('stat_features', np.zeros(100, dtype=np.float32)),
                    int(data['label'])
                )
            except Exception as e:
                print(f"[Warning] Failed to load cache for {file_path}: {e}")
        
        return None
    
    def _save_to_cache(self, file_path: Path, data: Tuple[np.ndarray, np.ndarray, np.ndarray, int]):
        """保存数据到缓存"""
        cache_path = self._get_cache_path(file_path)
        
        try:
            np.savez_compressed(
                cache_path,
                byte_sequence=data[0],
                pe_features=data[1],
                stat_features=data[2],
                label=data[3]
            )
        except Exception as e:
            print(f"[Warning] Failed to save cache for {file_path}: {e}")
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            Tuple of (byte_sequence, pe_features, stat_features, label)
            - byte_sequence: [max_byte_length] torch.uint8
            - pe_features: [pe_feature_dim] torch.float32
            - stat_features: [~100] torch.float32
            - label: torch.long
        """
        file_path = self.file_list[idx]
        label = self.label_list[idx]
        
        # 尝试从缓存加载
        if self.use_cache:
            cached_data = self._load_from_cache(file_path)
            if cached_data is not None:
                byte_seq, pe_feat, stat_feat, cached_label = cached_data
                if self.transform:
                    byte_seq, pe_feat, stat_feat = self.transform(byte_seq, pe_feat, stat_feat)
                if self.target_transform:
                    label = self.target_transform(label)
                return (
                    torch.from_numpy(byte_seq).long(),
                    torch.from_numpy(pe_feat).float(),
                    torch.from_numpy(stat_feat).float(),
                    torch.tensor(label, dtype=torch.long)
                )
        
        # 提取特征
        try:
            byte_seq, pe_feat, stat_feat, orig_len = extract_all_features(
                str(file_path), self.extraction_config
            )
            
            if byte_seq is None:
                raise ValueError("Feature extraction failed")
            
            # 填充/截断字节序列
            if len(byte_seq) > self.max_byte_length:
                byte_seq = byte_seq[:self.max_byte_length]
            elif len(byte_seq) < self.max_byte_length:
                byte_seq = np.pad(byte_seq, (0, self.max_byte_length - len(byte_seq)))
            
            # 确保 PE 特征维度正确
            if len(pe_feat) < self.pe_feature_dim:
                pe_feat = np.pad(pe_feat, (0, self.pe_feature_dim - len(pe_feat)))
            elif len(pe_feat) > self.pe_feature_dim:
                pe_feat = pe_feat[:self.pe_feature_dim]
            
            # 保存到缓存
            if self.use_cache:
                self._save_to_cache(file_path, (byte_seq, pe_feat, stat_feat, label))
            
            # 应用转换
            if self.transform:
                byte_seq, pe_feat, stat_feat = self.transform(byte_seq, pe_feat, stat_feat)
            if self.target_transform:
                label = self.target_transform(label)
            
            return (
                torch.from_numpy(byte_seq).long(),
                torch.from_numpy(pe_feat).float(),
                torch.from_numpy(stat_feat).float(),
                torch.tensor(label, dtype=torch.long)
            )
            
        except Exception as e:
            print(f"[Error] Failed to load {file_path}: {e}")
            # 返回零填充的默认值
            return (
                torch.zeros(self.max_byte_length, dtype=torch.long),
                torch.zeros(self.pe_feature_dim, dtype=torch.float32),
                torch.zeros(100, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long)
            )


class NPZDataset(Dataset):
    """预提取的 NPZ 数据集
    
    适用于已经使用 KVD 特征提取器处理过的数据。
    """
    
    def __init__(
        self,
        npz_dir: str,
        split: str = "train",  # train, val, test
        max_byte_length: int = 65536,
        pe_feature_dim: int = 1500,
    ):
        self.npz_dir = Path(npz_dir) / split
        self.max_byte_length = max_byte_length
        self.pe_feature_dim = pe_feature_dim
        
        # 收集所有 NPZ 文件
        self.npz_files = sorted(list(self.npz_dir.glob("*.npz")))
        
        if len(self.npz_files) == 0:
            raise ValueError(f"No NPZ files found in {self.npz_dir}")
    
    def __len__(self) -> int:
        return len(self.npz_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        npz_path = self.npz_files[idx]
        
        try:
            data = np.load(npz_path)
            
            byte_seq = data['byte_sequence']
            pe_feat = data['pe_features']
            label = int(data.get('label', 0))
            
            # 填充/截断
            if len(byte_seq) > self.max_byte_length:
                byte_seq = byte_seq[:self.max_byte_length]
            elif len(byte_seq) < self.max_byte_length:
                byte_seq = np.pad(byte_seq, (0, self.max_byte_length - len(byte_seq)))
            
            if len(pe_feat) < self.pe_feature_dim:
                pe_feat = np.pad(pe_feat, (0, self.pe_feature_dim - len(pe_feat)))
            elif len(pe_feat) > self.pe_feature_dim:
                pe_feat = pe_feat[:self.pe_feature_dim]
            
            # 统计特征（如果有）
            stat_feat = data.get('stat_features', np.zeros(100, dtype=np.float32))
            
            return (
                torch.from_numpy(byte_seq).long(),
                torch.from_numpy(pe_feat).float(),
                torch.from_numpy(stat_feat).float(),
                torch.tensor(label, dtype=torch.long)
            )
            
        except Exception as e:
            print(f"[Error] Failed to load {npz_path}: {e}")
            return (
                torch.zeros(self.max_byte_length, dtype=torch.long),
                torch.zeros(self.pe_feature_dim, dtype=torch.float32),
                torch.zeros(100, dtype=torch.float32),
                torch.tensor(0, dtype=torch.long)
            )


class NPZDataLoader:
    """NPZ 数据加载器封装
    
    提供便捷的数据加载接口。
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        max_byte_length: int = 65536,
        pe_feature_dim: int = 1500,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
    ):
        """
        Args:
            data_dir: 数据目录路径
            batch_size: 批次大小
            max_byte_length: 最大字节序列长度
            pe_feature_dim: PE 特征维度
            num_workers: 数据加载线程数
            pin_memory: 是否固定内存
            shuffle: 是否打乱数据
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_byte_length = max_byte_length
        self.pe_feature_dim = pe_feature_dim
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        
        self.dataset = None
        self.loader = None
    
    def create_dataloader(self, split: str = "train") -> DataLoader:
        """创建数据加载器
        
        Args:
            split: 数据集划分 (train, val, test)
        """
        try:
            # 尝试使用 NPZ 数据集
            dataset = NPZDataset(
                npz_dir=self.data_dir,
                split=split,
                max_byte_length=self.max_byte_length,
                pe_feature_dim=self.pe_feature_dim,
            )
        except ValueError:
            # 回退到原始文件数据集
            dataset = MalwareDataset(
                data_dir=self.data_dir,
                max_byte_length=self.max_byte_length,
                pe_feature_dim=self.pe_feature_dim,
            )
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle and split == "train",
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=split == "train",
        )
        
        return loader
    
    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        return self.create_dataloader("train")
    
    def get_val_loader(self) -> DataLoader:
        """获取验证数据加载器"""
        return self.create_dataloader("val")
    
    def get_test_loader(self) -> DataLoader:
        """获取测试数据加载器"""
        return self.create_dataloader("test")


def create_stratified_split(
    dataset: Dataset,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """创建分层划分的数据集
    
    Args:
        dataset: 完整数据集
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    np.random.seed(seed)
    
    # 按标签分层
    labels = dataset.label_list
    unique_labels = list(set(labels))
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label in unique_labels:
        label_indices = [i for i, l in enumerate(labels) if l == label]
        np.random.shuffle(label_indices)
        
        n = len(label_indices)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        
        val_indices.extend(label_indices[:n_val])
        test_indices.extend(label_indices[n_val:n_val + n_test])
        train_indices.extend(label_indices[n_val + n_test:])
    
    # 创建子数据集
    class SubDataset(Dataset):
        def __init__(self, base_dataset, indices):
            self.base_dataset = base_dataset
            self.indices = indices
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.base_dataset[self.indices[idx]]
    
    train_dataset = SubDataset(dataset, train_indices)
    val_dataset = SubDataset(dataset, val_indices)
    test_dataset = SubDataset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset
