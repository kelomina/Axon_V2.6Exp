#!/usr/bin/env python3
"""Axon v2.6 实验主脚本。

用于训练、评估和测试 Axon 恶意软件检测模型。

使用方法：
    # 训练
    python main.py train --data-dir data/samples --epochs 50
    
    # 评估
    python main.py eval --checkpoint models/best_model.pt --data-dir data/test
    
    # 测试单个文件
    python main.py predict --file path/to/sample.exe
    
    # 特征提取
    python main.py extract --data-dir raw_samples --output-dir data/extracted
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

import torch

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import AxonExperimentConfig, TrainingConfig
from model import AxonMalwareModel
from dataset import MalwareDataset, NPZDataLoader, create_stratified_split
from trainer import AxonTrainer
from kvd_features import extract_all_features, ExtractionConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Axon v2.6 Malware Detection Training and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data-dir', type=str, required=True,
                              help='Training data directory')
    train_parser.add_argument('--epochs', type=int, default=50,
                              help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=16,
                              help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4,
                              help='Learning rate')
    train_parser.add_argument('--device', type=str, default='cuda',
                              choices=['cuda', 'cpu'],
                              help='Device to use')
    train_parser.add_argument('--output-dir', type=str, default='models',
                              help='Output directory for models')
    train_parser.add_argument('--resume', type=str, default=None,
                              help='Resume from checkpoint')
    
    # 评估命令
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to model checkpoint')
    eval_parser.add_argument('--data-dir', type=str, required=True,
                            help='Evaluation data directory')
    eval_parser.add_argument('--batch-size', type=int, default=16,
                            help='Batch size')
    eval_parser.add_argument('--device', type=str, default='cuda',
                            choices=['cuda', 'cpu'],
                            help='Device to use')
    eval_parser.add_argument('--output', type=str, default='eval_report.json',
                            help='Output report file')
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='Predict on a file')
    predict_parser.add_argument('--file', type=str, required=True,
                               help='File to predict')
    predict_parser.add_argument('--checkpoint', type=str, required=True,
                               help='Path to model checkpoint')
    predict_parser.add_argument('--device', type=str, default='cuda',
                               choices=['cuda', 'cpu'],
                               help='Device to use')
    
    # 特征提取命令
    extract_parser = subparsers.add_parser('extract', help='Extract features')
    extract_parser.add_argument('--data-dir', type=str, required=True,
                               help='Input data directory')
    extract_parser.add_argument('--output-dir', type=str, required=True,
                               help='Output directory for extracted features')
    extract_parser.add_argument('--max-workers', type=int, default=4,
                               help='Number of parallel workers')
    
    return parser.parse_args()


def train_command(args):
    """训练命令"""
    print("=" * 60)
    print("Axon v2.6 Training")
    print("=" * 60)
    
    # 配置
    config = AxonExperimentConfig(
        experiment_name="axon_v2.6_train",
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        model_save_dir=args.output_dir,
    )
    
    train_config = TrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    # 创建模型
    print("\nInitializing model...")
    model = AxonMalwareModel(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 加载数据
    print("\nLoading data...")
    data_loader = NPZDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
    )
    
    try:
        train_loader = data_loader.get_train_loader()
        val_loader = data_loader.get_val_loader()
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        print("Creating dataset from raw files...")
        
        # 从原始文件创建数据集
        dataset = MalwareDataset(
            data_dir=args.data_dir,
            max_byte_length=config.max_byte_length,
            pe_feature_dim=config.pe_feature_dim,
        )
        
        # 分层划分
        train_dataset, val_dataset, _ = create_stratified_split(
            dataset, val_ratio=0.2, test_ratio=0.1
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    # 创建训练器
    trainer = AxonTrainer(model, config, train_config)
    
    # 恢复检查点（如果指定）
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))
    
    # 训练
    print("\nStarting training...")
    results = trainer.train(train_loader, val_loader)
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best F1 Score: {trainer.best_f1:.4f}")
    print(f"Best Epoch: {trainer.best_epoch}")
    print(f"Model saved to: {config.model_save_dir}")
    
    return results


def eval_command(args):
    """评估命令"""
    print("=" * 60)
    print("Axon v2.6 Evaluation")
    print("=" * 60)
    
    # 加载检查点
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    config = AxonExperimentConfig.from_dict(checkpoint['config'])
    
    # 创建模型
    model = AxonMalwareModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    # 加载数据
    print("\nLoading evaluation data...")
    data_loader = NPZDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    
    try:
        test_loader = data_loader.get_test_loader()
    except:
        dataset = MalwareDataset(
            data_dir=args.data_dir,
            max_byte_length=config.max_byte_length,
            pe_feature_dim=config.pe_feature_dim,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
    
    # 评估
    print("\nEvaluating...")
    trainer = AxonTrainer(model, config)
    results = trainer.evaluate(test_loader, 0, "test")
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Loss: {results.loss:.4f}")
    print(f"Accuracy: {results.accuracy:.4f}")
    print(f"Precision: {results.precision:.4f}")
    print(f"Recall: {results.recall:.4f}")
    print(f"F1 Score: {results.f1:.4f}")
    if results.auc:
        print(f"AUC: {results.auc:.4f}")
    
    # 保存报告
    import json
    report = {
        'loss': float(results.loss),
        'accuracy': float(results.accuracy),
        'precision': float(results.precision),
        'recall': float(results.recall),
        'f1': float(results.f1),
        'auc': float(results.auc) if results.auc else None,
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {output_path}")


def predict_command(args):
    """预测命令"""
    print("=" * 60)
    print("Axon v2.6 Prediction")
    print("=" * 60)
    
    # 检查文件
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"[Error] File not found: {file_path}")
        return
    
    # 加载检查点
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    config = AxonExperimentConfig.from_dict(checkpoint['config'])
    
    # 创建模型
    model = AxonMalwareModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    # 提取特征
    print("\nExtracting features...")
    extraction_config = ExtractionConfig(
        max_file_size=config.max_byte_length,
        pe_feature_dim=config.pe_feature_dim
    )
    
    byte_seq, pe_features, _, _ = extract_all_features(str(file_path), extraction_config)
    
    if byte_seq is None:
        print("[Error] Feature extraction failed")
        return
    
    # 转换为张量
    byte_tensor = torch.from_numpy(byte_seq).long().unsqueeze(0).to(args.device)
    pe_tensor = torch.from_numpy(pe_features).float().unsqueeze(0).to(args.device)
    
    # 预测
    print("Predicting...")
    with torch.no_grad():
        logits = model(byte_tensor, pe_tensor)['logits']
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
    
    # 输出结果
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"File: {file_path.name}")
    print(f"Prediction: {'Malicious' if pred == 1 else 'Benign'}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probabilities: Benign={probs[0, 0]:.4f}, Malicious={probs[0, 1]:.4f}")


def extract_command(args):
    """特征提取命令"""
    print("=" * 60)
    print("Axon v2.6 Feature Extraction")
    print("=" * 60)
    
    input_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"[Error] Input directory not found: {input_dir}")
        return
    
    # 统计
    total_files = 0
    success_files = 0
    failed_files = []
    
    # 遍历所有文件
    print(f"\nExtracting features from: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    extraction_config = ExtractionConfig(
        max_file_size=65536,
        pe_feature_dim=1500
    )
    
    for file_path in input_dir.rglob("*"):
        if not file_path.is_file():
            continue
        
        total_files += 1
        try:
            # 提取特征
            byte_seq, pe_features, stat_features, orig_len = extract_all_features(
                str(file_path), extraction_config
            )
            
            if byte_seq is None:
                raise ValueError("Feature extraction failed")
            
            # 保存到 NPZ
            output_file = output_dir / f"{file_path.stem}.npz"
            import numpy as np
            np.savez_compressed(
                output_file,
                byte_sequence=byte_seq,
                pe_features=pe_features,
                stat_features=stat_features,
                orig_length=orig_len
            )
            
            success_files += 1
            
            if total_files % 100 == 0:
                print(f"Processed: {total_files} files, Success: {success_files}")
        
        except Exception as e:
            failed_files.append((str(file_path), str(e)))
    
    # 打印统计
    print("\n" + "=" * 60)
    print("Extraction Complete")
    print("=" * 60)
    print(f"Total files: {total_files}")
    print(f"Success: {success_files}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for path, error in failed_files[:10]:
            print(f"  - {path}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")


def main():
    """主函数"""
    args = parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'eval':
        eval_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'extract':
        extract_command(args)
    else:
        print("Please specify a command. Use --help for usage information.")
        sys.exit(1)


if __name__ == '__main__':
    main()
