"""PGD（投影梯度下降）L∞ 无目标攻击。

参考 Madry 等人 "Towards Deep Learning Models Resistant to Adversarial
Attacks"（https://arxiv.org/abs/1706.06083）以及 torchattacks 的官方实现：
每一步沿交叉熵梯度符号上升，随机起点，最后投影回 epsilon 球和像素范围。
"""

import torch
import torch.nn as nn

from .base import project_linf, random_linf_start


class PGDAttack:
    """L∞ PGD 攻击。

    参数：
        model: 接收 [0, 1] 像素输入并返回 logits 的模型（通常为 NormalizedModel）。
        epsilon: L∞ 扰动上限，默认 4/255。
        steps: 迭代步数，默认 10。
        alpha: 每步步长；为 None 时按 epsilon / steps * alpha_scale 计算。
        alpha_scale: 自动步长倍率，默认 2.5。
        random_start: 是否在 epsilon 球内随机初始化，默认开启。
    """

    def __init__(self, model, epsilon=4 / 255, steps=10, alpha=None,
                 alpha_scale=2.5, random_start=True):
        if steps <= 0:
            raise ValueError('steps 必须为正整数')
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha if alpha is not None else epsilon / steps * alpha_scale
        self.random_start = random_start

    @torch.enable_grad()
    def attack(self, images, labels):
        """生成对抗样本，返回与输入同形状、取值在 [0, 1] 的张量。"""
        images = images.detach()
        labels = labels.detach()
        criterion = nn.CrossEntropyLoss()

        if self.random_start:
            adv = random_linf_start(images, self.epsilon)
        else:
            adv = images.clone()

        for _ in range(self.steps):
            adv.requires_grad_(True)
            loss = criterion(self.model(adv), labels)
            grad = torch.autograd.grad(loss, adv)[0]

            # 沿梯度符号走一步，再投影回合法扰动范围。
            adv = adv.detach() + self.alpha * grad.sign()
            adv = project_linf(adv, images, self.epsilon)

        return adv.detach()
