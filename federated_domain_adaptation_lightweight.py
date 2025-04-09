import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
import random
from models_lightweight import LightUNet, LightDomainDiscriminator, LightMMD_loss

class FederatedDomainAdaptationLightweight:
    """
    轻量级联邦学习框架，用于视网膜血管分割的域适应
    
    该类实现了一个联邦学习系统，多个客户端（数据集）协作训练全局模型，同时适应数据集之间的域偏移。
    
    关键特性：
    1. 使用轻量级模型架构减少显存占用
    2. 实现梯度累积以支持更大的等效批量大小
    3. 使用混合精度训练加速计算并减少显存需求
    4. 简化的域适应模块，保持核心功能
    5. 可配置的训练参数，以适应不同的硬件环境
    """
    def __init__(self, datasets, device, args):
        """
        初始化联邦学习系统
        
        Args:
            datasets (dict): 每个客户端的数据集字典
            device (torch.device): 运行模型的设备
            args (dict): 配置参数
        """
        self.datasets = datasets
        self.device = device
        self.args = args
        
        # 初始化混合精度训练
        self.use_amp = args.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 梯度累积步数
        self.grad_accumulation_steps = args.get('grad_accumulation_steps', 1)
        
        # 初始化全局模型
        self.global_model = LightUNet(
            n_channels=3,
            n_classes=1,
            with_domain_features=True
        ).to(device)
        
        # 初始化每个客户端的本地模型
        self.local_models = {}
        for client_id in datasets.keys():
            self.local_models[client_id] = LightUNet(
                n_channels=3,
                n_classes=1,
                with_domain_features=True
            ).to(device)
            # 使用全局模型权重初始化本地模型
            self.local_models[client_id].load_state_dict(self.global_model.state_dict())
        
        # 初始化域判别器
        self.domain_discriminator = LightDomainDiscriminator().to(device)
        
        # 初始化MMD损失用于域适应
        self.mmd_loss = LightMMD_loss().to(device)
        
        # 设置优化器
        self.global_optimizer = optim.Adam(self.global_model.parameters(), lr=args['lr'])
        
        self.local_optimizers = {}
        self.discriminator_optimizers = {}
        for client_id in datasets.keys():
            self.local_optimizers[client_id] = optim.Adam(
                self.local_models[client_id].parameters(), 
                lr=args['lr']
            )
            self.discriminator_optimizers[client_id] = optim.Adam(
                self.domain_discriminator.parameters(),
                lr=args['lr'] * 0.1  # 判别器使用较低的学习率
            )
        
        # 设置损失函数
        self.segmentation_loss = nn.BCEWithLogitsLoss()
        self.domain_loss = nn.CrossEntropyLoss()
        self.distillation_loss = nn.KLDivLoss(reduction='batchmean')
        
        # 跟踪指标
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_metrics': {}
        }
    
    def train(self, num_rounds):
        """
        训练联邦学习系统指定轮数
        
        Args:
            num_rounds (int): 联邦学习轮数
        
        Returns:
            dict: 训练指标
        """
        for round_idx in range(num_rounds):
            print(f"\n联邦学习轮次 {round_idx+1}/{num_rounds}")
            
            # 每个客户端的本地训练
            local_weights = {}
            local_losses = []
            
            for client_id in self.datasets.keys():
                print(f"训练客户端 {client_id}")
                
                # 训练本地模型
                local_loss = self.train_client(
                    client_id=client_id,
                    global_round=round_idx
                )
                local_losses.append(local_loss)
                
                # 获取模型权重
                local_weights[client_id] = self.local_models[client_id].state_dict()
            
            # 更新全局模型（联邦聚合）
            self.aggregate_models(local_weights, round_idx)
            
            # 评估全局模型
            if (round_idx + 1) % self.args['eval_every'] == 0:
                self.evaluate_global_model(round_idx)
            
            # 使用全局模型更新本地模型
            for client_id in self.datasets.keys():
                # 个性化模型更新：混合全局模型和本地模型
                self.personalized_model_update(client_id, round_idx)
        
        return self.metrics
    
    def train_client(self, client_id, global_round):
        """
        训练客户端的本地模型
        
        Args:
            client_id (str): 客户端标识符
            global_round (int): 当前全局轮次
        
        Returns:
            float: 平均训练损失
        """
        # 将模型设置为训练模式
        local_model = self.local_models[client_id]
        local_model.train()
        self.domain_discriminator.train()
        
        # 获取此客户端的数据加载器
        train_loader = self.datasets[client_id]['train']
        
        # 设置优化器
        optimizer = self.local_optimizers[client_id]
        discriminator_optimizer = self.discriminator_optimizers[client_id]
        
        epoch_loss = []
        for epoch in range(self.args['local_epochs']):
            batch_loss = []
            optimizer.zero_grad()  # 在梯度累积前清零梯度
            
            for batch_idx, data in enumerate(train_loader):
                images = data['image'].to(self.device)
                masks = data['mask'].to(self.device)
                domain_labels = data['domain'].to(self.device)
                
                # 计算梯度累积的缩放因子
                scale_factor = 1.0 / self.grad_accumulation_steps
                
                # 使用混合精度训练
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        # 前向传播
                        # Alpha控制梯度反转的强度（随训练增加）
                        alpha = min(1.0, (global_round * self.args['local_epochs'] + epoch) / (self.args['num_rounds'] * self.args['local_epochs'] / 2))
                        
                        # 从本地模型获取预测
                        pred_masks, domain_preds, features = local_model(images, alpha)
                        
                        # 分割损失
                        seg_loss = self.segmentation_loss(pred_masks, masks)
                        
                        # 域对抗损失
                        if domain_preds is not None:
                            domain_adv_loss = self.domain_loss(domain_preds, torch.argmax(domain_labels, dim=1))
                        else:
                            domain_adv_loss = torch.tensor(0.0).to(self.device)
                        
                        # 从全局模型进行知识蒸馏（如果不是第一轮）
                        if global_round > 0:
                            # 获取全局模型的预测
                            with torch.no_grad():
                                self.global_model.eval()
                                global_pred_masks, _, global_features = self.global_model(images)
                            
                            # 特征级蒸馏损失
                            distill_loss = F.mse_loss(features, global_features.detach())
                            
                            # 本地和全局特征之间的MMD损失
                            mmd = self.mmd_loss(features, global_features.detach())
                        else:
                            distill_loss = torch.tensor(0.0).to(self.device)
                            mmd = torch.tensor(0.0).to(self.device)
                        
                        # 总损失
                        loss = seg_loss + \
                               self.args['lambda_adv'] * domain_adv_loss + \
                               self.args['lambda_distill'] * distill_loss + \
                               self.args['lambda_mmd'] * mmd
                        
                        # 缩放损失以适应梯度累积
                        loss = loss * scale_factor
                    
                    # 反向传播和优化
                    self.scaler.scale(loss).backward()
                    
                    # 梯度累积
                    if (batch_idx + 1) % self.grad_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad()
                else:
                    # 前向传播
                    alpha = min(1.0, (global_round * self.args['local_epochs'] + epoch) / (self.args['num_rounds'] * self.args['local_epochs'] / 2))
                    pred_masks, domain_preds, features = local_model(images, alpha)
                    
                    # 分割损失
                    seg_loss = self.segmentation_loss(pred_masks, masks)
                    
                    # 域对抗损失
                    if domain_preds is not None:
                        domain_adv_loss = self.domain_loss(domain_preds, torch.argmax(domain_labels, dim=1))
                    else:
                        domain_adv_loss = torch.tensor(0.0).to(self.device)
                    
                    # 从全局模型进行知识蒸馏（如果不是第一轮）
                    if global_round > 0:
                        # 获取全局模型的预测
                        with torch.no_grad():
                            self.global_model.eval()
                            global_pred_masks, _, global_features = self.global_model(images)
                        
                        # 特征级蒸馏损失
                        distill_loss = F.mse_loss(features, global_features.detach())
                        
                        # 本地和全局特征之间的MMD损失
                        mmd = self.mmd_loss(features, global_features.detach())
                    else:
                        distill_loss = torch.tensor(0.0).to(self.device)
                        mmd = torch.tensor(0.0).to(self.device)
                    
                    # 总损失
                    loss = seg_loss + \
                           self.args['lambda_adv'] * domain_adv_loss + \
                           self.args['lambda_distill'] * distill_loss + \
                           self.args['lambda_mmd'] * mmd
                    
                    # 缩放损失以适应梯度累积
                    loss = loss * scale_factor
                    
                    # 反向传播和优化
                    loss.backward()
                    
                    # 梯度累积
                    if (batch_idx + 1) % self.grad_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()
                
                batch_loss.append(loss.item() / scale_factor)  # 存储未缩放的损失值
                
                # 单独训练域判别器
                if global_round > 0:  # 第一轮后开始域适应
                    discriminator_optimizer.zero_grad()
                    
                    # 提取特征，不进行梯度反转
                    with torch.no_grad():
                        _, _, features = local_model(images)
                    
                    # 域分类
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            domain_preds = self.domain_discriminator(features.view(features.size(0), -1).detach())
                            d_loss = self.domain_loss(domain_preds, torch.argmax(domain_labels, dim=1))
                        
                        self.scaler.scale(d_loss).backward()
                        self.scaler.step(discriminator_optimizer)
                        self.scaler.update()
                    else:
                        domain_preds = self.domain_discriminator(features.view(features.size(0), -1).detach())
                        d_loss = self.domain_loss(domain_preds, torch.argmax(domain_labels, dim=1))
                        
                        d_loss.backward()
                        discriminator_optimizer.step()
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print(f"客户端 {client_id} - 本地轮次 {epoch+1}/{self.args['local_epochs']} - 损失: {epoch_loss[-1]:.4f}")
        
        return sum(epoch_loss) / len(epoch_loss)
    
    def aggregate_models(self, local_weights, global_round):
        """
        使用加权平均聚合本地模型到全局模型
        
        Args:
            local_weights (dict): 本地模型权重字典
            global_round (int): 当前全局轮次
        """
        # 基于数据集大小计算聚合权重
        total_samples = sum([len(self.datasets[client_id]['train'].dataset) for client_id in self.datasets.keys()])
        client_weights = {client_id: len(self.datasets[client_id]['train'].dataset) / total_samples 
                         for client_id in self.datasets.keys()}
        
        # 使用零初始化全局模型字典
        global_dict = OrderedDict()
        for k in local_weights[list(local_weights.keys())[0]].keys():
            global_dict[k] = torch.zeros_like(local_weights[list(local_weights.keys())[0]][k])
        
        # 模型参数的加权平均
        for client_id in local_weights.keys():
            weight = client_weights[client_id]
            for k in global_dict.keys():
                global_dict[k] += local_weights[client_id][k] * weight
        
        # 更新全局模型
        self.global_model.load_state_dict(global_dict)
    
    def personalized_model_update(self, client_id, global_round):
        """
        使用全局和本地参数的个性化混合更新本地模型
        
        Args:
            client_id (str): 客户端标识符
            global_round (int): 当前全局轮次
        """
        # 自适应混合参数（随轮次增加以偏向全局知识）
        beta = min(0.8, 0.1 + global_round * 0.1)  # 最大0.8以保留一些本地知识
        
        # 获取全局和本地模型状态
        global_dict = self.global_model.state_dict()
        local_dict = self.local_models[client_id].state_dict()
        
        # 创建混合模型
        mixed_dict = OrderedDict()
        for k in global_dict.keys():
            mixed_dict[k] = beta * global_dict[k] + (1 - beta) * local_dict[k]
        
        # 更新本地模型
        self.local_models[client_id].load_state_dict(mixed_dict)
    
    def evaluate_global_model(self, global_round):
        """
        在所有客户端的测试数据上评估全局模型
        
        Args:
            global_round (int): 当前全局轮次
        """
        self.global_model.eval()
        
        val_loss = 0
        dice_scores = {}
        
        with torch.no_grad():
            for client_id in self.datasets.keys():
                test_loader = self.datasets[client_id]['test']
                client_loss = 0
                client_dice = 0
                
                for batch_idx, data in enumerate(test_loader):
                    images = data['image'].to(self.device)
                    masks = data['mask'].to(self.device)
                    
                    # 前向传播
                    pred_masks, _, _ = self.global_model(images)
                    
                    # 计算损失
                    loss = self.segmentation_loss(pred_masks, masks)
                    client_loss += loss.item()
                    
                    # 计算Dice分数
                    pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                    dice = (2 * (pred_binary * masks).sum()) / ((pred_binary + masks).sum() + 1e-8)
                    client_dice += dice.item()
                
                # 平均指标
                client_loss /= len(test_loader)
                client_dice /= len(test_loader)
                
                val_loss += client_loss
                dice_scores[client_id] = client_dice
                
                print(f"客户端 {client_id} - 测试损失: {client_loss:.4f}, Dice分数: {client_dice:.4f}")
        
        # 所有客户端的平均验证损失
        val_loss /= len(self.datasets)
        
        # 存储指标
        self.metrics['val_loss'].append(val_loss)
        self.metrics['test_metrics'][global_round] = {
            'dice_scores': dice_scores,
            'avg_dice': sum(dice_scores.values()) / len(dice_scores)
        }
        
        print(f"全局轮次 {global_round+1} - 平均测试损失: {val_loss:.4f}, 平均Dice: {self.metrics['test_metrics'][global_round]['avg_dice']:.4f}")
    
    def save_models(self, save_path):
        """
        保存全局和本地模型
        
        Args:
            save_path (str): 保存模型的目录
        """
        os.makedirs(save_path, exist_ok=True)
        
        # 保存全局模型
        torch.save(self.global_model.state_dict(), os.path.join(save_path, 'global_model.pth'))
        
        # 保存本地模型
        for client_id in self.local_models.keys():
            torch.save(
                self.local_models[client_id].state_dict(),
                os.path.join(save_path, f'local_model_{client_id}.pth')
            )
    
    def load_models(self, load_path):
        """
        加载全局和本地模型
        
        Args:
            load_path (str): 加载模型的目录
        """
        # 加载全局模型
        self.global_model.load_state_dict(torch.load(os.path.join(load_path, 'global_model.pth')))
        
        # 加载本地模型
        for client_id in self.local_models.keys():
            model_path = os.path.join(load_path, f'local_model_{client_id}.pth')
            if os.path.exists(model_path):
                self.local_models[client_id].load_state_dict(torch.load(model_path))
            else:
                print(f"客户端 {client_id} 的本地模型未找到，使用全局模型初始化")
                self.local_models[client_id].load_state_dict(self.global_model.state_dict())