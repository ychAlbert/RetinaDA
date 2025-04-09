import os
import argparse
import torch
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt

from data_loader_lightweight import get_data_loaders_lightweight
from models_lightweight import LightUNet
from federated_domain_adaptation_lightweight import FederatedDomainAdaptationLightweight
from evaluation import SegmentationEvaluator

def set_seed(seed):
    """
    设置随机种子以确保可重现性
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='轻量级联邦域适应学习用于视网膜血管分割')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='./RentinaDA', help='数据集目录路径')
    parser.add_argument('--datasets', type=str, nargs='+', default=['DRIVE', 'STARE', 'CHASDB', 'HRF', 'LES-AV', 'RAVIR'], 
                        help='要使用的数据集列表')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=2, help='训练的批量大小')
    parser.add_argument('--num_rounds', type=int, default=50, help='联邦学习轮数')
    parser.add_argument('--local_epochs', type=int, default=3, help='每轮的本地训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--eval_every', type=int, default=5, help='每N轮评估一次全局模型')
    
    # 模型参数
    parser.add_argument('--lambda_adv', type=float, default=0.05, help='对抗损失权重')
    parser.add_argument('--lambda_distill', type=float, default=0.3, help='蒸馏损失权重')
    parser.add_argument('--lambda_mmd', type=float, default=0.05, help='MMD损失权重')
    
    # 优化参数
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--grad_accumulation_steps', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--img_size', type=int, default=384, help='图像大小')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU ID')
    parser.add_argument('--save_dir', type=str, default='./results', help='保存结果的目录')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称')
    
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建实验目录
    if args.exp_name is None:
        args.exp_name = f"fedda_lightweight_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据集
    print("加载数据集...")
    data_loaders = get_data_loaders_lightweight(
        base_path=args.data_path,
        dataset_names=args.datasets,
        batch_size=args.batch_size,
        num_workers=2,  # 减少工作线程数
        augment=True,
        img_size=(args.img_size, args.img_size)  # 使用较小的图像尺寸
    )
    
    # 创建联邦学习配置
    fed_args = {
        'lr': args.lr,
        'num_rounds': args.num_rounds,
        'local_epochs': args.local_epochs,
        'eval_every': args.eval_every,
        'lambda_adv': args.lambda_adv,
        'lambda_distill': args.lambda_distill,
        'lambda_mmd': args.lambda_mmd,
        'use_amp': args.use_amp,  # 混合精度训练
        'grad_accumulation_steps': args.grad_accumulation_steps  # 梯度累积
    }
    
    # 初始化联邦学习系统
    print("初始化轻量级联邦学习系统...")
    fed_system = FederatedDomainAdaptationLightweight(
        datasets=data_loaders,
        device=device,
        args=fed_args
    )
    
    # 训练联邦学习系统
    print("开始联邦学习训练...")
    metrics = fed_system.train(args.num_rounds)
    
    # 保存模型
    print("保存模型...")
    fed_system.save_models(save_dir)
    
    # 评估最终模型
    print("\n评估最终模型...")
    evaluator = SegmentationEvaluator(device)
    
    # 评估全局模型
    print("\n评估全局模型:")
    global_results = evaluator.evaluate_all_datasets(
        model=fed_system.global_model,
        datasets=data_loaders
    )
    
    # 保存全局模型结果
    evaluator.plot_metrics(
        results=global_results,
        save_path=os.path.join(save_dir, 'global_model_metrics.png')
    )
    
    # 评估本地模型
    local_results = {}
    for client_id in data_loaders.keys():
        print(f"\n评估{client_id}的本地模型:")
        local_model = fed_system.local_models[client_id]
        
        # 评估所有数据集以测量泛化能力
        client_results = evaluator.evaluate_all_datasets(
            model=local_model,
            datasets=data_loaders
        )
        
        local_results[client_id] = client_results
        
        # 保存分割结果以进行视觉检查
        evaluator.save_segmentation_results(
            model=local_model,
            data_loader=data_loaders[client_id]['test'],
            save_dir=os.path.join(save_dir, f'segmentation_results_{client_id}')
        )
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['val_loss'], label='验证损失')
    plt.xlabel('评估轮次')
    plt.ylabel('损失')
    plt.title('训练期间的验证损失')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    
    # 绘制Dice分数随轮次变化
    plt.figure(figsize=(12, 6))
    for client_id in data_loaders.keys():
        dice_scores = [metrics['test_metrics'][round]['dice_scores'][client_id] 
                      for round in metrics['test_metrics'].keys()]
        plt.plot(list(metrics['test_metrics'].keys()), dice_scores, label=client_id)
    
    plt.xlabel('轮次')
    plt.ylabel('Dice分数')
    plt.title('训练期间的Dice分数')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'dice_scores.png'))
    
    print(f"\n训练完成。结果保存到 {save_dir}")

if __name__ == '__main__':
    main()