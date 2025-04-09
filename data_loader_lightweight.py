import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

class RetinalDatasetLightweight(Dataset):
    """
    轻量级视网膜血管分割数据集类
    减小图像尺寸和内存占用
    """
    def __init__(self, dataset_path, dataset_name, is_train=True, transform=None, augment=False, img_size=(384, 384)):
        """
        Args:
            dataset_path (str): 数据集目录路径
            dataset_name (str): 数据集名称 (DRIVE, STARE, CHASEDB, HRF, LES-AV, RAVIR)
            is_train (bool): 是否使用训练集
            transform (callable, optional): 可选的图像变换
            augment (bool): 是否使用数据增强
            img_size (tuple): 调整图像大小的目标尺寸 (height, width)，默认减小到384x384
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.is_train = is_train
        self.transform = transform
        self.augment = augment
        self.img_size = img_size
        
        # 根据数据集名称定义路径
        if dataset_name == 'DRIVE':
            self.image_dir = os.path.join(dataset_path, 'DRIVE', 'images')
            self.mask_dir = os.path.join(dataset_path, 'DRIVE', '1st_manual')
        elif dataset_name == 'STARE':
            self.image_dir = os.path.join(dataset_path, 'STARE', 'images')
            self.mask_dir = os.path.join(dataset_path, 'STARE', '1st_manual')
        elif dataset_name == 'CHASDB':
            self.image_dir = os.path.join(dataset_path, 'CHASDB', 'images')
            self.mask_dir = os.path.join(dataset_path, 'CHASDB', '1st_manual')
        elif dataset_name == 'HRF':
            self.image_dir = os.path.join(dataset_path, 'HRF', 'images')
            self.mask_dir = os.path.join(dataset_path, 'HRF', 'manual1')
        elif dataset_name == 'LES-AV':
            self.image_dir = os.path.join(dataset_path, 'LES-AV', 'images')
            self.mask_dir = os.path.join(dataset_path, 'LES-AV', 'vessel-segmentations')
        elif dataset_name == 'RAVIR':
            self.image_dir = os.path.join(dataset_path, 'RAVIR', 'training_images')
            self.mask_dir = os.path.join(dataset_path, 'RAVIR', 'training_masks')
        else:
            raise ValueError(f"不支持数据集 {dataset_name}")
        
        # 获取所有图像文件
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
        # 分割训练集和测试集（如果没有预定义，则使用80/20分割）
        if is_train:
            self.image_files = self.image_files[:int(len(self.image_files) * 0.8)]
        else:
            self.image_files = self.image_files[int(len(self.image_files) * 0.8):]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 对于掩码，我们需要找到相应的掩码文件
        # 命名约定可能因数据集而异
        mask_name = img_name
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 如果掩码文件不存在，尝试找到匹配项
        if not os.path.exists(mask_path):
            # 从图像名称中提取基本信息
            # 处理不同的命名模式
            if '_test' in img_name or '_training' in img_name:
                # 对于像'10_test_3.png'这样的文件，提取'10'作为基本编号
                base_num = img_name.split('_')[0]
                # 查找具有模式'XX_manual1.png'或'XX_manual1_Y.png'的相应掩码
                potential_masks = [f for f in os.listdir(self.mask_dir) 
                                 if f.startswith(base_num + '_manual1')]
            else:
                # 对于其他命名模式，使用原始方法
                base_name = os.path.splitext(img_name)[0]
                potential_masks = [f for f in os.listdir(self.mask_dir) if base_name in f]
                
            if potential_masks:
                mask_name = potential_masks[0]
                mask_path = os.path.join(self.mask_dir, mask_name)
            else:
                raise FileNotFoundError(f"未找到与{img_name}匹配的掩码")
        
        # 加载图像和掩码
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 确保掩码是二值的（0或1）
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        
        # 调整图像和掩码大小到相同尺寸
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 如果启用，应用数据增强
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # 转换为张量
        image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        # 如果提供，应用额外的变换
        if self.transform:
            image = self.transform(image)
        
        # 创建域标签（独热编码）
        domain_idx = {'DRIVE': 0, 'STARE': 1, 'CHASDB': 2, 'HRF': 3, 'LES-AV': 4, 'RAVIR': 5}[self.dataset_name]
        domain_label = torch.zeros(6)
        domain_label[domain_idx] = 1
        
        return {
            'image': image,
            'mask': mask,
            'domain': domain_label,
            'dataset_name': self.dataset_name,
            'image_path': img_path
        }
    
    def _augment(self, image, mask):
        """
        对图像和掩码应用数据增强
        简化版本，减少计算量
        """
        # 随机水平翻转
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # 随机垂直翻转
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        # 随机旋转（角度范围减小）
        angle = random.randint(-10, 10)  # 减小旋转角度范围
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        mask = cv2.warpAffine(mask, M, (w, h))
        
        # 随机亮度和对比度调整（减小范围）
        alpha = 1.0 + random.uniform(-0.1, 0.1)  # 对比度，减小范围
        beta = random.uniform(-10, 10)  # 亮度，减小范围
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return image, mask

def get_data_loaders_lightweight(base_path, dataset_names, batch_size=2, num_workers=2, augment=True, img_size=(384, 384)):
    """
    为多个数据集创建轻量级数据加载器
    
    Args:
        base_path (str): 数据集的基本路径
        dataset_names (list): 要包含的数据集名称列表
        batch_size (int): 数据加载器的批量大小，默认减小到2
        num_workers (int): 数据加载的工作线程数，默认减小到2
        augment (bool): 是否使用数据增强
        img_size (tuple): 调整图像大小的目标尺寸 (height, width)，默认384x384
    
    Returns:
        dict: 包含每个数据集的训练和测试数据加载器的字典
    """
    data_loaders = {}
    
    for dataset_name in dataset_names:
        # 创建训练数据集
        train_dataset = RetinalDatasetLightweight(
            dataset_path=base_path,
            dataset_name=dataset_name,
            is_train=True,
            augment=augment,
            img_size=img_size
        )
        
        # 创建测试数据集
        test_dataset = RetinalDatasetLightweight(
            dataset_path=base_path,
            dataset_name=dataset_name,
            is_train=False,
            augment=False,
            img_size=img_size
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # 丢弃最后一个不完整的批次以避免批次大小不一致
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # 将数据加载器添加到字典
        data_loaders[dataset_name] = {
            'train': train_loader,
            'test': test_loader
        }
    
    return data_loaders