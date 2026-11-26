"""频域低通滤波，用于构造原图的低频参考图像。

输入是 [0, 1] 像素空间的 [N, C, H, W] 张量；滤波在频域对幅度谱乘一个
随距中心频率距离衰减的掩膜，再逆变换回像素空间。这与 CSR 检测阶段
“比较原图与低通滤波图像嵌入”的做法一致。
"""

import torch

FILTER_TYPES = ('gaussian', 'butterworth', 'ideal')


def lowpass_mask(height, width, radius, filter_type='gaussian', order=2,
                 device=None, dtype=torch.float32):
    """生成频域低通掩膜，返回 [H, W]，取值 [0, 1]。

    radius 是衰减带宽半径，单位是频谱中心（直流分量）之外的像素/频点距离：
      * gaussian    exp(-d² / (2r²))，平滑衰减，中心恒为 1；
      * butterworth 1 / (1 + (d/r)^(2·order))，阶数越高越接近理想滤波；
      * ideal       d <= r 为 1，否则为 0。
    """
    if radius <= 0:
        raise ValueError(f'低通半径必须为正数，实际为 {radius}')
    if filter_type not in FILTER_TYPES:
        raise ValueError(f'不支持的低通类型 {filter_type!r}，可选 {", ".join(FILTER_TYPES)}')
    if filter_type == 'butterworth' and order <= 0:
        raise ValueError(f'巴特沃斯阶数必须为正数，实际为 {order}')

    y = torch.arange(height, device=device, dtype=dtype) - height // 2
    x = torch.arange(width, device=device, dtype=dtype) - width // 2
    dist_sq = y[:, None] ** 2 + x[None, :] ** 2

    if filter_type == 'gaussian':
        mask = torch.exp(-dist_sq / (2 * radius ** 2))
    elif filter_type == 'butterworth':
        mask = 1.0 / (1.0 + (torch.sqrt(dist_sq) / radius) ** (2 * order))
    else:
        mask = (dist_sq <= radius ** 2).to(dtype)
    return mask


def apply_lowpass(images, radius, filter_type='gaussian', order=2):
    """对 [N, C, H, W]、取值 [0, 1] 的图像做低通滤波，返回同形状张量。"""
    if images.dim() != 4:
        raise ValueError(f'期望 [N, C, H, W] 输入，实际为 {tuple(images.shape)}')

    height, width = images.shape[-2:]
    mask = lowpass_mask(height, width, radius, filter_type, order,
                        device=images.device, dtype=images.dtype)
    spectrum = torch.fft.fftshift(torch.fft.fft2(images), dim=(-2, -1))
    filtered = torch.fft.ifft2(
        torch.fft.ifftshift(spectrum * mask.view(1, 1, height, width), dim=(-2, -1)))
    return torch.clamp(filtered.real, 0.0, 1.0)


def filter_spectrum(spectrum, radius, filter_type='gaussian', order=2):
    """对已经 fftshift 的频谱应用低通掩膜；便于同一批图像复用 FFT。"""
    height, width = spectrum.shape[-2:]
    mask = lowpass_mask(height, width, radius, filter_type, order,
                        device=spectrum.device, dtype=spectrum.real.dtype)
    return spectrum * mask.view(1, 1, height, width)


def inverse_spectrum(spectrum):
    """把 fftshift 后的频谱逆变换回 [0, 1] 像素空间。"""
    return torch.clamp(
        torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1))).real, 0.0, 1.0)
