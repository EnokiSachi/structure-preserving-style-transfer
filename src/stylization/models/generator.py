import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """SE注意力模块（本科毕设创新点：提升特征提取能力）"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """残差块（解决深度网络退化问题）"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            SEBlock(channels)  # 加入SE注意力模块
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """生成器：将内容图转换为风格迁移图"""
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # 下采样：压缩特征
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # 残差块：提取深度特征
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            # 上采样：恢复图像尺寸
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # 输出层：生成3通道RGB图像
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()  # 输出值限制在[-1,1]
        )

    def forward(self, x):
        return self.model(x)