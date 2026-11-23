"""对抗攻击集合。

所有攻击统一约定：
    * 输入图像为 [N, C, H, W] 的像素张量，取值 [0, 1]（通常 C=1 的灰度图）；
    * 模型由 NormalizedModel 包装，内部完成 ImageNet 归一化；
    * 调用 attack(images, labels) 返回同形状、同取值范围的对抗样本。
"""

from .apgd import APGDAttack
from .base import NormalizedModel
from .bim import BIMAttack
from .cw import CWAttack
from .pgd import PGDAttack

ATTACKS = {
    'pgd': PGDAttack,
    'bim': BIMAttack,
    'cw': CWAttack,
    'apgd': APGDAttack,
}

__all__ = [
    'NormalizedModel',
    'PGDAttack',
    'BIMAttack',
    'CWAttack',
    'APGDAttack',
    'ATTACKS',
    'build_attack',
]


def build_attack(name, model, **params):
    """按名称创建攻击实例，params 会原样传给对应攻击的构造函数。"""
    try:
        attack_class = ATTACKS[name]
    except KeyError:
        raise ValueError(f'未知攻击 {name!r}，可选 {", ".join(ATTACKS)}') from None
    return attack_class(model, **params)
