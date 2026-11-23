"""CW（Carlini-Wagner）L2 无目标攻击。

参考 Carlini 与 Wagner "Towards Evaluating the Robustness of Neural Networks"
（https://arxiv.org/abs/1608.04644）以及 torchattacks 的官方实现：
用 tanh 空间保证像素始终在 [0, 1] 内，以 Adam 同时优化 L2 扰动和
CW 边界损失 f(x) = max(max_{i!=y} z_i - z_y, -kappa)，并记录满足
误分类条件且 L2 最小的样本。该攻击是 L2 范数，不使用 epsilon 预算。
"""

import torch
import torch.nn as nn


class CWAttack:
    """L2 CW 攻击。

    参数：
        model: 接收 [0, 1] 像素输入并返回 logits 的模型。
        c: 边界损失的权重，越大越倾向成功攻击，默认 1.0。
        kappa: 置信度边界，默认 0；成功判断条件为 max_other - true > kappa。
        steps: Adam 迭代步数，默认 50。
        lr: Adam 学习率，默认 0.01。
    """

    def __init__(self, model, c=1.0, kappa=0.0, steps=50, lr=0.01):
        if steps <= 0:
            raise ValueError('steps 必须为正整数')
        self.model = model
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    @staticmethod
    def _atanh(x):
        """稳定的反双曲正切，避免像素恰好为 0 或 1 时出现 inf。"""
        x = x.clamp(-1 + 1e-6, 1 - 1e-6)
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _margin_loss(self, logits, labels):
        """CW 的 f 函数：拉大真实类与其他类最大 logit 的差距。"""
        real = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        other_logits = logits.clone()
        other_logits.scatter_(1, labels.view(-1, 1), -float('inf'))
        other = other_logits.max(dim=1)[0]
        return torch.clamp(real - other, min=-self.kappa)

    @torch.enable_grad()
    def attack(self, images, labels):
        """生成对抗样本，返回与输入同形状、取值在 [0, 1] 的张量。"""
        images = images.detach()
        labels = labels.detach()
        batch = images.shape[0]

        # 像素 [0,1] -> tanh 空间 [-1,1]，优化变量为 w。
        w = self._atanh(images * 2 - 1).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=self.lr)

        best_adv = images.clone()
        best_l2 = torch.full((batch,), 1e10, device=images.device,
                             dtype=images.dtype)
        prev_cost = 1e10

        for step in range(self.steps):
            adv = 0.5 * (torch.tanh(w) + 1)
            l2 = (adv - images).flatten(1).pow(2).sum(dim=1)

            logits = self.model(adv)
            cost = l2.sum() + self.c * self._margin_loss(logits, labels).sum()

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 只保留“已误分类且 L2 更小”的样本，避免越优化越差。
            with torch.no_grad():
                misclassified = logits.argmax(dim=1) != labels
                better = misclassified & (best_l2 > l2.detach())
                best_l2 = torch.where(better, l2.detach(), best_l2)
                best_adv[better] = adv.detach()[better]

            # 与官方实现一致的早停：损失不再下降就返回当前最优结果。
            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv
                prev_cost = cost.item()

        return best_adv
