"""APGD（Auto-PGD，APGD-CE）L∞ 无目标攻击。

参考 Croce 与 Hein "Reliable evaluation of adversarial robustness with an
ensemble of diverse parameter-free attacks"（https://arxiv.org/abs/2003.01690）
以及 autoattack 官方实现的核心流程：
    * 随机起点；
    * 按交叉熵损失保留历史最优点；
    * 动量式更新（上一轮位移参与混合）；
    * 每经过若干步检查损失是否振荡，对振荡样本把步长减半并回退到最优点。
"""

import torch
import torch.nn as nn

from .base import project_linf


class APGDAttack:
    """L∞ APGD-CE 攻击。

    参数：
        model: 接收 [0, 1] 像素输入并返回 logits 的模型。
        epsilon: L∞ 扰动上限，默认 4/255。
        steps: 迭代步数，默认 30。
        rho: 振荡判据（最近 k 步中损失上升比例不超过 rho 即视为振荡），默认 0.75。
    """

    def __init__(self, model, epsilon=4 / 255, steps=30, rho=0.75):
        if steps <= 0:
            raise ValueError('steps 必须为正整数')
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.rho = rho

    @torch.enable_grad()
    def attack(self, images, labels):
        """生成对抗样本，返回与输入同形状、取值在 [0, 1] 的张量。"""
        images = images.detach()
        labels = labels.detach()
        batch = images.shape[0]
        epsilon = self.epsilon
        criterion = nn.CrossEntropyLoss(reduction='none')

        # 随机起点：均匀采样后逐样本归一化到 L∞ 半径 epsilon。
        noise = 2 * torch.rand_like(images) - 1
        scale = noise.flatten(1).abs().max(dim=1)[0].clamp_min(1e-12)
        scale = scale.view(-1, 1, 1, 1)
        x_adv = torch.clamp(images + epsilon * noise / scale, 0.0, 1.0)

        # 初始梯度、损失与历史最优点。
        x_tmp = x_adv.detach().requires_grad_(True)
        loss_indiv = criterion(self.model(x_tmp), labels)
        grad = torch.autograd.grad(loss_indiv.sum(), x_tmp)[0].detach()
        loss_best = loss_indiv.detach().clone()
        grad_best = grad.clone()
        x_best = x_adv.detach().clone()

        step_size = torch.full((batch, 1, 1, 1), 2.0 * epsilon,
                               device=images.device, dtype=images.dtype)
        x_adv = x_adv.detach()
        x_adv_old = x_adv.clone()

        # 与官方实现一致的步长缩减参数。
        k = max(int(0.22 * self.steps), 1)       # 振荡检查窗口
        k_min = max(int(0.06 * self.steps), 1)   # 窗口下限
        size_decr = max(int(0.03 * self.steps), 1)
        counter = 0
        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        loss_history = [loss_best.clone()]

        for iteration in range(self.steps):
            with torch.no_grad():
                # 上一轮位移，用作动量项。
                displacement = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                # 首轮不混合历史位移，之后按 0.75 混合。
                momentum = 0.75 if iteration > 0 else 1.0
                x_next = project_linf(
                    x_adv + step_size * grad.sign(), images, epsilon)
                x_adv = project_linf(
                    x_adv + (x_next - x_adv) * momentum
                    + displacement * (1.0 - momentum),
                    images, epsilon)

            # 在新位置重新计算梯度与逐样本损失。
            x_tmp = x_adv.detach().requires_grad_(True)
            loss_indiv = criterion(self.model(x_tmp), labels)
            grad = torch.autograd.grad(loss_indiv.sum(), x_tmp)[0].detach()
            loss_now = loss_indiv.detach()
            x_adv = x_adv.detach()

            with torch.no_grad():
                # 损失越大越好，记录历史最优。
                better = loss_now > loss_best
                x_best[better] = x_adv[better]
                grad_best[better] = grad[better]
                loss_best[better] = loss_now[better]
                loss_history.append(loss_now.clone())

                # 到达检查窗口时判断是否振荡或长时间没有提升。
                counter += 1
                if counter == k:
                    recent = torch.stack(loss_history[-k - 1:])
                    increases = (recent[1:] > recent[:-1]).sum(dim=0).float()
                    oscillating = increases <= self.rho * k
                    no_improve = (1.0 - reduced_last_check) * (
                        loss_best_last_check >= loss_best).float()
                    reduce_mask = torch.maximum(oscillating.float(), no_improve)

                    reduced_last_check = reduce_mask.clone()
                    loss_best_last_check = loss_best.clone()

                    if reduce_mask.gt(0).any():
                        indices = reduce_mask.gt(0).nonzero(as_tuple=True)[0]
                        step_size[indices] = step_size[indices] / 2.0
                        x_adv[indices] = x_best[indices]
                        grad[indices] = grad_best[indices]

                    k = max(k - size_decr, k_min)
                    counter = 0

        return x_best.detach()
