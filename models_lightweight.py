import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积，用于减少参数量和计算量"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LightConv(nn.Module):
    """轻量级卷积块，替代标准的双卷积块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class LightDown(nn.Module):
    """轻量级下采样模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            LightConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class LightUp(nn.Module):
    """轻量级上采样模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = LightConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配的情况
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 连接特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class LightUNet(nn.Module):
    """轻量级UNet模型，减少通道数和层数"""
    def __init__(self, n_channels=3, n_classes=1, with_domain_features=False):
        super(LightUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.with_domain_features = with_domain_features
        
        # 减少通道数以降低内存占用
        base_channels = 32  # 原始UNet使用64
        
        self.inc = LightConv(n_channels, base_channels)
        self.down1 = LightDown(base_channels, base_channels*2)
        self.down2 = LightDown(base_channels*2, base_channels*4)
        self.down3 = LightDown(base_channels*4, base_channels*8)
        
        # 移除最深层以减少参数量
        self.up1 = LightUp(base_channels*8, base_channels*4)
        self.up2 = LightUp(base_channels*4, base_channels*2)
        self.up3 = LightUp(base_channels*2, base_channels)
        self.outc = OutConv(base_channels, n_classes)
        
        # 简化的域分类器
        if with_domain_features:
            self.domain_classifier = nn.Sequential(
                nn.Linear(base_channels*8 * 64 * 64, 256),  # 假设特征图大小为64x64
                nn.ReLU(True),
                nn.Dropout(0.3),  # 减少dropout率以加速训练
                nn.Linear(256, 6)  # 6个域
            )

    def forward(self, x, alpha=0):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 存储瓶颈特征用于域适应
        bottleneck_features = x4
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        
        # 域分类（如果启用）
        domain_preds = None
        if self.with_domain_features and self.training:
            # 应用梯度反转层进行对抗训练
            reversed_features = GradientReversalLayer.apply(bottleneck_features, alpha)
            domain_preds = self.domain_classifier(reversed_features.view(reversed_features.size(0), -1))
        
        return logits, domain_preds, bottleneck_features

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class LightDomainDiscriminator(nn.Module):
    """轻量级域判别器"""
    def __init__(self, input_dim=256, hidden_dim=128):
        super(LightDomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 6)  # 6个域
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 减少dropout率
    
    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.layer2(x)
        return x

class LightMMD_loss(nn.Module):
    """简化的MMD损失函数"""
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=3):  # 减少kernel_num
        super(LightMMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.kernel_type = kernel_type
        self.fix_sigma = None
        
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=3, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target):
        batch_size = source.size()[0]
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss