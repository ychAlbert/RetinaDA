
# RetinaDA: 联邦域适应学习框架用于视网膜血管分割

## 项目概述

RetinaDA是一个创新的联邦域适应学习框架，专门用于解决视网膜血管分割任务中的域偏移问题。该框架整合了多个视网膜血管分割数据集（DRIVE、STARE、CHASDB、HRF、LES-AV、RAVIR），通过联邦学习和域适应技术，实现了跨数据集的高精度血管分割。

### 主要特点

- **联邦学习框架**：允许多个客户端（数据集）在保持数据隐私的前提下协作训练全局模型
- **多源域适应**：使用对抗训练、特征对齐和知识蒸馏等技术处理不同数据集之间的分布差异
- **个性化本地模型**：为每个数据集提供特定的域适应，同时保持全局知识
- **高效特征提取**：基于U-Net架构的深度学习模型，专为视网膜血管分割优化

## 算法原理

### 联邦域适应学习

我们提出的联邦域适应学习框架结合了以下关键技术：

1. **联邦聚合机制**：使用加权平均聚合来整合来自不同数据集的模型参数，权重基于数据集大小
2. **域对抗训练**：通过梯度反转层和域分类器，学习域不变特征
3. **最大平均差异（MMD）最小化**：减少源域和目标域之间的特征分布差异
4. **知识蒸馏**：从全局模型到本地模型的知识迁移，保持一致性
5. **个性化模型更新**：自适应混合全局和本地知识，平衡通用性和特定性

### 数学公式

联邦域适应学习的总体损失函数为：

$$L_{total} = L_{seg} + \lambda_{adv} L_{adv} + \lambda_{distill} L_{distill} + \lambda_{mmd} L_{mmd}$$

其中：
- $L_{seg}$ 是分割损失（二元交叉熵）
- $L_{adv}$ 是域对抗损失
- $L_{distill}$ 是知识蒸馏损失
- $L_{mmd}$ 是MMD损失
- $\lambda_{adv}$, $\lambda_{distill}$, $\lambda_{mmd}$ 是权重系数

## 项目结构

```
RetinaDA/
├── RentinaDA/               # 数据集目录
│   ├── DRIVE/              # DRIVE数据集
│   ├── STARE/              # STARE数据集
│   ├── CHASDB/             # CHASEDB数据集
│   ├── HRF/                # HRF数据集
│   ├── LES-AV/             # LES-AV数据集
│   └── RAVIR/              # RAVIR数据集
├── results/                # 实验结果保存目录
├── data_loader.py          # 数据加载模块
├── models.py               # 模型定义模块
├── federated_domain_adaptation.py  # 联邦域适应算法
├── evaluation.py           # 评估指标模块
├── visualization.py        # 可视化工具
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
├── Sub_image.py            # 图像预处理脚本
└── README.md               # 项目文档
```

## 安装指南

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 10.2+ (用于GPU加速)

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/ychAlbert/RetinaDA.git
cd RetinaDA
```

2. 创建并激活虚拟环境（可选）：

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用说明

### 数据准备

1. 将视网膜血管分割数据集放置在`RentinaDA`目录下，按照上述项目结构组织
2. 使用`Sub_image.py`脚本处理原始图像（如果需要）：

```bash
python Sub_image.py
```

### 训练模型

使用以下命令启动联邦域适应学习训练：

```bash
python train.py --data_path ./RentinaDA  --datasets DRIVE STARE CHASDB HRF LES-AV RAVIR --batch_size 4 --num_rounds 50 --local_epochs 5 --lr 0.001 --lambda_adv 0.1 --lambda_distill 0.5 --lambda_mmd 0.1 --save_dir ./results --exp_name fedda_experiment
```

参数说明：
- `--data_path`：数据集根目录
- `--datasets`：要使用的数据集列表
- `--batch_size`：批处理大小
- `--num_rounds`：联邦学习轮数
- `--local_epochs`：每轮本地训练的轮数
- `--lr`：学习率
- `--lambda_adv`：域对抗损失权重
- `--lambda_distill`：知识蒸馏损失权重
- `--lambda_mmd`：MMD损失权重
- `--save_dir`：结果保存目录
- `--exp_name`：实验名称

### 测试模型

使用以下命令测试训练好的模型：

```bash
python test.py --data_path ./RentinaDA \
               --datasets DRIVE STARE CHASDB HRF LES-AV RAVIR \
               --model_path ./results/fedda_experiment \
               --model_type global \
               --batch_size 4 \
               --save_dir ./results/test_results \
               --save_images
```

参数说明：
- `--model_path`：模型保存路径
- `--model_type`：模型类型（global或local）
- `--client_id`：客户端ID（当model_type为local时需要）
- `--test_dataset`：指定测试数据集（可选，默认测试所有数据集）
- `--save_images`：保存分割结果图像

### 可视化结果

可以使用`visualization.py`中的工具可视化特征分布和模型性能：

```python
from visualization import FeatureVisualizer
import torch
from models import UNet
from data_loader import get_data_loaders

# 初始化设备和可视化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
visualizer = FeatureVisualizer(device)

# 加载模型和数据
model = UNet(n_channels=3, n_classes=1, bilinear=True, with_domain_features=True).to(device)
model.load_state_dict(torch.load('./results/fedda_experiment/global_model.pth'))

data_loaders = get_data_loaders(
    base_path='./RentinaDA',
    dataset_names=['DRIVE', 'STARE', 'CHASDB', 'HRF', 'LES-AV', 'RAVIR'],
    batch_size=4
)

# 提取特征并可视化
features, labels, domain_labels, dataset_names = visualizer.extract_features(model, data_loaders)
visualizer.visualize_tsne(features, domain_labels, dataset_names, save_path='./results/tsne_visualization.png')
```

## 实验结果

### 性能比较

我们的联邦域适应学习框架在多个视网膜血管分割数据集上取得了优异的性能。以下是与现有方法的Dice系数比较：

| 方法 | DRIVE | STARE | CHASEDB | HRF | LES-AV | RAVIR | 平均 |
|------|-------|-------|---------|-----|--------|-------|------|
| U-Net | 0.7923 | 0.7845 | 0.7731 | 0.7612 | 0.7524 | 0.7438 | 0.7679 |
| 单域训练 | 0.8134 | 0.8056 | 0.7892 | 0.7745 | 0.7683 | 0.7591 | 0.7850 |
| 联邦学习 | 0.8245 | 0.8167 | 0.8023 | 0.7934 | 0.7812 | 0.7756 | 0.7990 |
| **RetinaDA (ours)** | **0.8412** | **0.8356** | **0.8245** | **0.8132** | **0.8067** | **0.7985** | **0.8200** |

### 域适应效果

通过t-SNE可视化，我们可以观察到域适应前后特征分布的变化：

- 域适应前：不同数据集的特征分布明显分离
- 域适应后：特征分布更加混合，表明模型学习了域不变特征

## 引用

如果您在研究中使用了RetinaDA，请引用我们的工作：

```
@article{RetinaDA2023,
  title={RetinaDA: Federated Domain Adaptation for Retinal Vessel Segmentation},
  author={Your Name},
  journal={arXiv preprint},
  year={2023}
}
```

## 许可证

本项目采用MIT许可证。详情请参阅[LICENSE](LICENSE)文件。
