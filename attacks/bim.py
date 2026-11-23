"""BIM / I-FGSM（基本迭代法）L∞ 无目标攻击。

参考 Kurakin 等人 "Adversarial Examples in the Physical World"
（https://arxiv.org/abs/1607.02533）以及 torchattacks 的官方实现：
从干净图像出发，每步按交叉熵梯度符号累加 alpha，并把总扰动裁剪到
epsilon 以内。与 PGD 的区别主要是不做随机初始化。
"""

import torch
import torch.nn as nn

from .base import project_linf


class BIMAttack:
    """L∞ BIM（I-FGSM）攻击。

    参数：
        model: 接收 [0, 1] 像素输入并返回 logits 的模型。
        epsilon: L∞ 扰动上限，默认 4/255。
        steps: 迭代步数，默认 10。
        alpha: 每步步长；为 None 时取 epsilon / steps（I-FGSM 的常用设置）。
        alpha_scale: 自动步长倍率，默认 1.0。
    """

    def __init__(self, model, epsilon=4 / 255, steps=10, alpha=None,
                 alpha_scale=1.0):
        if steps <= 0:
            raise ValueError('steps 必须为正整数')
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha if alpha is not None else epsilon / steps * alpha_scale

    @torch.enable_grad()
    def attack(self, images, labels):
        """生成对抗样本，返回与输入同形状、取值在 [0, 1] 的张量。"""
        images = images.detach()
        labels = labels.detach()
        criterion = nn.CrossEntropyLoss()

        adv = images.clone()
        for _ in range(self.steps):
            adv.requires_grad_(True)
            loss = criterion(self.model(adv), labels)
            grad = torch.autograd.grad(loss, adv)[0]

            adv = adv.detach() + self.alpha * grad.sign()
            adv = project_linf(adv, images, self.epsilon)

        return adv.detach()
