"""位平面分解与噪声块选择，移植自 LOTA 的 bit_patch.py。

LOTA（ICCV 2025）用低位平面放大图像的微弱噪声：把每个像素的低 k 位取出
并线性拉伸到 0~255，再在噪声图上挑选局部差分能量最大的 patch 送入分类器。
原实现工作在单张 uint8 RGB 图像上（learn_from/LOTA/bit_patch.py）；这里改写
为批量张量接口，输入 [N, C, H, W]、取值 [0, 1]，输出同一形状，便于接在
data.dataset 的像素空间之后、ImageNet 归一化之前。

与 LOTA 的对应关系：
    bit_mode='scaling'       低位线性拉伸，保留噪声幅度（LOTA 默认）
    bit_mode='thresholding'  低位二值化；LOTA 仓库此处只留了未实现的占位，
                             这里按 2^(k-1) 做阈值补全
    patch_mode='max'         差分能量最大的块（LOTA 评估默认）
    patch_mode='min'         差分能量最小的块
    patch_mode='random'      随机块（LOTA 训练默认）

典型用法（把 LOTA 式噪声图作为模型输入）：
    from defense.bitplane import bitplane_noise
    noise = bitplane_noise(images, bits=3, patch_size=32, patch_mode='max')
    model_input = (noise - mean) / std      # 再做 ImageNet 归一化
"""

import torch
import torch.nn.functional as F

BIT_MODES = ('scaling', 'thresholding')
PATCH_MODES = ('max', 'min', 'random')


def quantize(images):
    """[0, 1] 浮点图 -> 0..255 整数，与 PIL 的 8 位量化保持一致。"""
    return torch.round(images.clamp(0.0, 1.0) * 255.0).to(torch.int32)


def _check_bits(bits):
    if not isinstance(bits, int) or not 1 <= bits <= 8:
        raise ValueError(f'bits 必须是 1~8 的整数，实际为 {bits!r}')


def low_bit_plane(images, bits=3, mode='scaling'):
    """取低 bits 位平面，返回与输入同形状、取值 [0, 1] 的噪声图。

    scaling 与 LOTA 完全一致：``(x & (2^bits - 1)) * (255 // (2^bits - 1))``，
    把 k 位取值拉伸回 8 位灰度范围；thresholding 则按 2^(k-1) 二值化。
    """
    _check_bits(bits)
    if mode not in BIT_MODES:
        raise ValueError(f'不支持的位平面模式 {mode!r}，可选 {", ".join(BIT_MODES)}')

    levels = (1 << bits) - 1
    low = quantize(images) & levels
    if mode == 'scaling':
        return low.to(images.dtype) * (255 // levels) / 255.0
    return (low >= (1 << (bits - 1))).to(images.dtype)


def bit_planes(images, bits=8):
    """返回 [N, bits, C, H, W] 的 0/1 位平面，第 0 个是 LSB。"""
    _check_bits(bits)
    quantized = quantize(images)
    return torch.stack(
        [((quantized >> shift) & 1).to(images.dtype) for shift in range(bits)],
        dim=1)


def bit_plane_energy(images, bits=8):
    """每个位平面中 1 的比例，返回 [N, bits]；第 0 列对应 LSB。"""
    return bit_planes(images, bits).mean(dim=(2, 3, 4))


def patch_score(patches):
    """LOTA 的 compute：块内水平、垂直、两条对角线相邻像素绝对差之和。

    patches 形状为 [..., C, h, w]，返回去掉最后三维后的逐块分数。
    """
    if patches.shape[-2] < 2 or patches.shape[-1] < 2:
        raise ValueError(f'patch 至少需要 2×2，实际为 {tuple(patches.shape[-2:])}')

    horizontal = (patches[..., :, :-1] - patches[..., :, 1:]).abs()
    vertical = (patches[..., :-1, :] - patches[..., 1:, :]).abs()
    diagonal = ((patches[..., :-1, :-1] - patches[..., 1:, 1:]).abs()
                + (patches[..., 1:, :-1] - patches[..., :-1, 1:]).abs())
    return (horizontal.sum(dim=(-1, -2, -3))
            + vertical.sum(dim=(-1, -2, -3))
            + diagonal.sum(dim=(-1, -2, -3)))


def tile_patches(images, patch_size):
    """非重叠切块，返回 [N, K, C, p, p]；右、下不足一块的边缘会被丢弃。"""
    height, width = images.shape[-2:]
    if patch_size < 1:
        raise ValueError(f'patch_size 必须为正数，实际为 {patch_size}')
    if patch_size > min(height, width):
        raise ValueError(f'patch_size={patch_size} 超过图像尺寸 {height}×{width}')

    tiles = images.unfold(-2, patch_size, patch_size).unfold(-1, patch_size, patch_size)
    channels = images.shape[1]
    return tiles.permute(0, 2, 3, 1, 4, 5).reshape(
        images.shape[0], -1, channels, patch_size, patch_size)


def random_patches(images, patch_size, num_patches, generator=None):
    """LOTA 的 RandomCrop：每张图随机裁 num_patches 个可重叠的块。

    位置在 CPU 上采样，所以 generator 应是 CPU 的 torch.Generator；为 None
    时使用全局随机数。返回 [N, num_patches, C, p, p]。
    """
    if num_patches < 1:
        raise ValueError(f'num_patches 必须为正数，实际为 {num_patches}')
    height, width = images.shape[-2:]
    if patch_size > min(height, width):
        raise ValueError(f'patch_size={patch_size} 超过图像尺寸 {height}×{width}')

    n, channels = images.shape[:2]
    ys = torch.randint(0, height - patch_size + 1, (n, num_patches),
                       generator=generator)
    xs = torch.randint(0, width - patch_size + 1, (n, num_patches),
                       generator=generator)
    ys, xs = ys.to(images.device), xs.to(images.device)

    offsets = torch.arange(patch_size, device=images.device)
    patches = images.new_empty(n, num_patches, channels, patch_size, patch_size)
    for index in range(n):
        index_y = ys[index, :, None] + offsets[None, :]        # [K, p]
        index_x = xs[index, :, None] + offsets[None, :]        # [K, p]
        rows = images[index, :, index_y, :]                    # [C, K, p, W]
        gather_index = index_x.view(1, num_patches, 1, patch_size).expand(
            channels, num_patches, patch_size, patch_size)
        patches[index] = rows.gather(3, gather_index).permute(1, 0, 2, 3)
    return patches


def extract_patches(images, patch_size=32, num_patches=None, generator=None):
    """num_patches 为 None 时非重叠切块，否则按 LOTA 方式随机裁块。"""
    if num_patches is None:
        return tile_patches(images, patch_size)
    return random_patches(images, patch_size, num_patches, generator)


def select_patch_index(scores, mode='max', generator=None):
    """按 LOTA 规则为每张图选择一个 patch，scores 为 [N, K]，返回 [N]。"""
    if mode not in PATCH_MODES:
        raise ValueError(f'不支持的选块模式 {mode!r}，可选 {", ".join(PATCH_MODES)}')
    if scores.dim() != 2:
        raise ValueError(f'期望 [N, K] 的分数表，实际为 {tuple(scores.shape)}')

    if mode == 'max':
        return scores.argmax(dim=1)
    if mode == 'min':
        return scores.argmin(dim=1)
    index = torch.randint(0, scores.shape[1], (scores.shape[0],), generator=generator)
    return index.to(scores.device)


def select_noise_patch(images, patch_size=32, mode='max', num_patches=None,
                       generator=None, resize=True):
    """在噪声图上选块；resize=True 时双线性放大回原图尺寸，便于送入模型。"""
    patches = extract_patches(images, patch_size, num_patches, generator)
    scores = patch_score(patches)
    index = select_patch_index(scores, mode, generator)
    selected = patches[torch.arange(patches.shape[0], device=patches.device), index]
    if resize:
        selected = F.interpolate(selected, size=images.shape[-2:],
                                 mode='bilinear', align_corners=False)
    return selected


def bitplane_noise(images, bits=3, bit_mode='scaling', patch_size=32,
                   patch_mode='max', num_patches=None, generator=None):
    """LOTA 完整流程：低位平面噪声图 -> 差分能量选块 -> 放大回原尺寸。"""
    noise = low_bit_plane(images, bits, bit_mode)
    return select_noise_patch(noise, patch_size, patch_mode, num_patches,
                              generator, resize=True)
