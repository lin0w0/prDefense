"""攻击脚本共用的模型包装与投影工具。

本项目的模型输入是 ImageNet 归一化后的三通道图像，而攻击通常在
[0, 1] 像素空间里进行，方便直接保存成图片。NormalizedModel 负责在
前向时补上归一化，攻击代码只关心像素值。
"""

import torch
import torch.nn as nn

from data.dataset import IMAGENET_MEAN, IMAGENET_STD


class NormalizedModel(nn.Module):
    """把 [0, 1] 像素输入归一化后送入分类模型。

    单通道灰度输入会复制成三通道，与训练时 Grayscale(num_output_channels=3)
    的预处理保持一致；攻击只需对输出求输入梯度，因此冻结网络参数以节省显存。
    """

    def __init__(self, model, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.model = model
        self.register_buffer(
            'mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def forward(self, images):
        if images.dim() != 4:
            raise ValueError(f'期望 [N, C, H, W] 输入，实际为 {tuple(images.shape)}')
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] != 3:
            raise ValueError(f'输入需为 1 或 3 通道，实际为 {images.shape[1]}')
        return self.model((images - self.mean) / self.std)


def random_linf_start(images, epsilon):
    """在 L∞ 球内均匀随机采样，并保证结果仍在 [0, 1] 像素范围内。"""
    perturbed = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    return torch.clamp(perturbed, 0.0, 1.0).detach()


def project_linf(adv, images, epsilon):
    """投影回 {|adv-images| <= epsilon} 与 [0, 1] 的交集。"""
    delta = torch.clamp(adv - images, -epsilon, epsilon)
    return torch.clamp(images + delta, 0.0, 1.0).detach()
