"""多尺度、多方向 Gabor 滤波器组与竞争方向编码。

本模块面向掌纹这类以线状纹理为主的灰度图：用一组不同波长 λ 的 Gabor
滤波器提取主线、褶皱和细粒度纹理的局部方向响应，再用 CompCode 的
“竞争”规则得到方向编码。所有函数都是批量张量接口，输入输出沿用项目
其它模块的约定：图像为 [N, C, H, W]、取值 [0, 1]。

方向约定
--------
``orientation`` 指 Gabor 滤波器的**载波方向**（正弦调制的传播方向，单位
度，周期 180°）。滤波器对沿该方向变化的条纹响应最强，因此对一条方向为
θ+90° 的纹线，载波方向为 θ 的滤波器会给出强响应。竞争编码取响应**最小**
的滤波器（CompCode 式最小响应规则）时，被选中的载波方向恰好等于纹线方向；取
响应最强时，被选中的载波方向是纹线法向。两种语义相差 90°，解读角度时
需要留意当前使用的是哪条规则。

尺度约定
--------
``wavelength`` λ 是载波的空间周期（像素）。λ 越大，滤波器越宽、对应越粗的
结构；默认的 4/6/8/12/16/24 大致覆盖细粒度纹理、褶皱和掌纹主线。空间
包络标准差为 ``sigma_ratio * λ``（默认 0.56λ，约一个倍频程带宽），沿载波
方向窄、垂直载波方向宽（由 ``gamma`` 控制长宽比）。

频域实现
--------
Gabor 核是高斯包络调制的复正弦，其频域响应是一个中心在
``(cosθ/λ, sinθ/λ)`` 的高斯。模块直接在 FFT 域乘以该单边高斯，再逆变换，
得到解析 Gabor 响应；实部/虚部分别对应偶/奇对称滤波器。相比空间域卷积，
多尺度大核的代价与核尺寸无关，适合一次算完整个滤波器组。图像先做反射
填充，避免 FFT 的循环边界把纹理绕回图像另一侧。

典型用法：
    from defense.gabor import (make_gabor_bank, gabor_magnitude,
                               dominant_orientation, orientation_histogram)

    bank = make_gabor_bank(224, 224, wavelengths=[4, 8, 16, 24],
                           orientations=6, device='cuda')
    magnitude = gabor_magnitude(images, bank)        # [N, S*O, H, W]
    codes = dominant_orientation(magnitude, bank)    # [N, S, H, W]
    hist = orientation_histogram(codes, bank, grid=4)  # [N, S, G, G, O]
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

DEFAULT_WAVELENGTHS = (4.0, 6.0, 8.0, 12.0, 16.0, 24.0)
DEFAULT_ORIENTATIONS = 6
DOMINANCE_RULES = ('min', 'max')


@dataclass(frozen=True)
class GaborBank:
    """一个多尺度、多方向 Gabor 滤波器组及其索引信息。"""

    masks: torch.Tensor           # [K, Hp, Wp]，频域高斯，K = S * O
    wavelengths: torch.Tensor     # [K]，每个滤波器对应的波长 λ
    orientations: torch.Tensor    # [K]，载波方向，单位度
    num_scales: int
    num_orientations: int
    pad: int                      # 反射填充宽度
    height: int
    width: int
    padded_height: int
    padded_width: int

    @property
    def num_filters(self):
        return self.masks.shape[0]

    @property
    def scale_angles(self):
        """每个尺度使用的方向列表，形状 [O]。"""
        return self.orientations.reshape(self.num_scales,
                                         self.num_orientations)[0]


def _orientation_values(orientations):
    """把方向数或显式角度列表统一成 [O] 的 float64 张量，取值 [0, 180)。"""
    if isinstance(orientations, int):
        if orientations < 1:
            raise ValueError(f'方向数必须为正整数，实际为 {orientations}')
        return torch.arange(orientations, dtype=torch.float64) * (180.0 / orientations)

    values = torch.as_tensor(orientations, dtype=torch.float64).flatten()
    if values.numel() == 0:
        raise ValueError('方向列表不能为空')
    values = torch.remainder(values, 180.0)
    if torch.unique(values).numel() != values.numel():
        raise ValueError('方向列表在 180° 周期内存在重复值')
    return values


def make_gabor_bank(height, width, wavelengths=DEFAULT_WAVELENGTHS,
                    orientations=DEFAULT_ORIENTATIONS, sigma_ratio=0.56,
                    gamma=0.5, pad_ratio=3.0, device=None,
                    dtype=torch.float32):
    """构造多尺度多方向 Gabor 滤波器组的频域掩膜。

    参数
    ----
    height, width : 输入图像尺寸（未填充前）。
    wavelengths   : 载波空间周期 λ（像素），可为标量或序列。
    orientations  : 方向数（等分 180°）或显式角度序列（度）。
    sigma_ratio   : 空间包络标准差 / λ，默认 0.56（约一个倍频程带宽）。
    gamma         : 包络长宽比，默认 0.5；越小，沿纹线方向拉得越长。
    pad_ratio     : 反射填充宽度 = pad_ratio × 最大包络标准差。
    """
    if height < 2 or width < 2:
        raise ValueError(f'图像尺寸至少为 2×2，实际为 {height}×{width}')
    wavelengths = torch.as_tensor(wavelengths, dtype=torch.float64).flatten()
    if wavelengths.numel() == 0:
        raise ValueError('波长列表不能为空')
    if torch.any(wavelengths <= 2.0):
        raise ValueError('波长必须大于 2 像素（否则会越过奈奎斯特频率）')
    if sigma_ratio <= 0:
        raise ValueError(f'sigma_ratio 必须为正数，实际为 {sigma_ratio}')
    if not 0 < gamma <= 1:
        raise ValueError(f'gamma 必须在 (0, 1] 内，实际为 {gamma}')
    if pad_ratio < 0:
        raise ValueError(f'pad_ratio 不能为负数，实际为 {pad_ratio}')

    angles = _orientation_values(orientations)
    num_scales, num_orientations = wavelengths.numel(), angles.numel()
    sigma = sigma_ratio * wavelengths                      # 沿载波方向的空间标准差
    pad = int(math.ceil(pad_ratio * float(sigma.max()) / gamma))
    if pad >= min(height, width):
        raise ValueError(f'填充宽度 {pad} 不小于图像尺寸，请减小 lambda 或 pad_ratio')
    padded_height, padded_width = height + 2 * pad, width + 2 * pad

    # 归一化频率（cycles/pixel），范围 [-0.5, 0.5)；无需 fftshift。
    freq_y = torch.fft.fftfreq(padded_height, dtype=torch.float64).view(-1, 1)
    freq_x = torch.fft.fftfreq(padded_width, dtype=torch.float64).view(1, -1)

    masks = []
    mask_wavelengths = []
    mask_orientations = []
    for index in range(num_scales):
        wavelength = float(wavelengths[index])
        sigma_a = 1.0 / (2.0 * math.pi * float(sigma[index]))  # 径向频宽
        sigma_b = gamma * sigma_a                               # 切向频宽
        for angle in angles.tolist():
            theta = math.radians(angle)
            # 方向以 x 轴（列方向）为 0°，逆时针为正。
            center_x = math.cos(theta) / wavelength
            center_y = math.sin(theta) / wavelength
            delta_x = freq_x - center_x
            delta_y = freq_y - center_y
            along = delta_x * math.cos(theta) + delta_y * math.sin(theta)
            across = -delta_x * math.sin(theta) + delta_y * math.cos(theta)
            mask = torch.exp(-along ** 2 / (2 * sigma_a ** 2)
                             - across ** 2 / (2 * sigma_b ** 2))
            # 去掉直流泄漏：等价于在空间域减去核的均值，使常值图像响应为零。
            # 不做这一步时，每个滤波器对图像均值都有约 0.2 的响应，会淹没
            # 不同方向之间的差异。
            mask[0, 0] = 0
            masks.append(mask)
            mask_wavelengths.append(wavelength)
            mask_orientations.append(angle)

    return GaborBank(
        masks=torch.stack(masks).to(device=device, dtype=dtype),
        wavelengths=torch.tensor(mask_wavelengths, dtype=torch.float32),
        orientations=torch.tensor(mask_orientations, dtype=torch.float32),
        num_scales=num_scales,
        num_orientations=num_orientations,
        pad=pad,
        height=height,
        width=width,
        padded_height=padded_height,
        padded_width=padded_width,
    )


def _as_grayscale(images):
    """统一成 [N, 1, H, W]；三通道灰度复制图取平均不会损失信息。"""
    if images.dim() == 3:
        images = images.unsqueeze(1)
    if images.dim() != 4:
        raise ValueError(f'期望 [N, C, H, W] 输入，实际为 {tuple(images.shape)}')
    if images.shape[1] == 1:
        return images
    return images.mean(dim=1, keepdim=True)


def _check_bank(images, bank):
    if images.shape[-2:] != (bank.height, bank.width):
        raise ValueError(
            f'图像尺寸 {tuple(images.shape[-2:])} 与滤波器组 '
            f'{bank.height}×{bank.width} 不一致')


@torch.no_grad()
def gabor_magnitude(images, bank, filter_chunk=8):
    """计算每个滤波器的解析 Gabor 响应幅度，返回 [N, K, H, W]。

    K = num_scales × num_orientations，顺序为尺度优先（先遍历波长，再遍历
    方向）。filter_chunk 控制一次逆变换的滤波器数量，仅影响显存占用。
    """
    images = _as_grayscale(images).to(device=bank.masks.device,
                                      dtype=bank.masks.dtype)
    _check_bank(images, bank)
    if filter_chunk is None or filter_chunk < 1:
        filter_chunk = bank.num_filters

    padded = F.pad(images, (bank.pad,) * 4, mode='reflect')
    spectrum = torch.fft.fft2(padded)                      # [N, 1, Hp, Wp]
    output = images.new_empty(images.shape[0], bank.num_filters,
                              bank.height, bank.width)
    top, left = bank.pad, bank.pad
    bottom, right = top + bank.height, left + bank.width

    for start in range(0, bank.num_filters, filter_chunk):
        stop = min(start + filter_chunk, bank.num_filters)
        response = torch.fft.ifft2(
            spectrum * bank.masks[start:stop][None, :, :, :])
        output[:, start:stop] = response[..., top:bottom, left:right].abs()
    return output


@torch.no_grad()
def gabor_response(images, bank, filter_chunk=8):
    """返回解析 Gabor 复响应 [N, K, H, W]（实部/虚部为偶/奇滤波器响应）。"""
    images = _as_grayscale(images).to(device=bank.masks.device,
                                      dtype=bank.masks.dtype)
    _check_bank(images, bank)
    if filter_chunk is None or filter_chunk < 1:
        filter_chunk = bank.num_filters

    padded = F.pad(images, (bank.pad,) * 4, mode='reflect')
    spectrum = torch.fft.fft2(padded)
    top, left = bank.pad, bank.pad
    bottom, right = top + bank.height, left + bank.width
    chunks = []
    for start in range(0, bank.num_filters, filter_chunk):
        stop = min(start + filter_chunk, bank.num_filters)
        response = torch.fft.ifft2(
            spectrum * bank.masks[start:stop][None, :, :, :])
        chunks.append(response[..., top:bottom, left:right])
    return torch.cat(chunks, dim=1)


def dominant_orientation(magnitude, bank, rule='min'):
    """按竞争规则选取每个像素的主方向，返回 [N, S, H, W] 的编码。

    ``rule='min'`` 采用 CompCode 式的最小响应规则（本文用解析 Gabor 幅度
    实现）：取响应最小的滤波器，其载波方向等于该像素的纹线方向。
    ``rule='max'`` 取响应最强的滤波器，其载波方向是纹线
    法向；两种规则是同一滤波器组上的不同竞争判决，方向语义相差 90°，分析
    稳定性时可在同一数据上对比，但解释具体角度时需要区分。
    """
    if rule not in DOMINANCE_RULES:
        raise ValueError(f'不支持的主方向规则 {rule!r}，可选 {", ".join(DOMINANCE_RULES)}')
    if magnitude.dim() != 4:
        raise ValueError(f'期望 [N, S*O, H, W] 输入，实际为 {tuple(magnitude.shape)}')
    n, k, height, width = magnitude.shape
    if k != bank.num_scales * bank.num_orientations:
        raise ValueError(f'滤波器数 {k} 与 bank 的 '
                         f'{bank.num_scales * bank.num_orientations} 不一致')
    scores = magnitude.reshape(n, bank.num_scales, bank.num_orientations, -1)
    index = scores.argmin(dim=2) if rule == 'min' else scores.argmax(dim=2)
    return index.reshape(n, bank.num_scales, height, width)


def orientation_histogram(codes, bank, grid=4, eps=1e-8):
    """把方向编码变成分块方向直方图，返回 [N, S, G, G, O]。

    每个 G×G 块内统计各方向编码的出现次数，再对每个块的 O 维直方图做
    L2 归一化，使不同块对后续余弦相似度的贡献相当。grid 不能超过图像边长。
    """
    if codes.dim() != 4:
        raise ValueError(f'期望 [N, S, H, W] 编码输入，实际为 {tuple(codes.shape)}')
    if codes.shape[1] != bank.num_scales:
        raise ValueError('编码的尺度数与 bank 不一致')
    height, width = codes.shape[-2:]
    if grid < 1 or grid > min(height, width):
        raise ValueError(f'grid={grid} 超出图像尺寸 {tuple(height, width)}')

    # 每个像素映射到 G×G 网格中的一个块；块编号 × O + 方向码 作为一维下标。
    rows = torch.arange(height, device=codes.device) * grid // height
    cols = torch.arange(width, device=codes.device) * grid // width
    block = rows[:, None] * grid + cols[None, :]           # [H, W]
    flat_codes = codes.reshape(-1, height, width)          # [N*S, H, W]
    index = (block[None, :, :] * bank.num_orientations + flat_codes).reshape(
        flat_codes.shape[0], -1)
    counts = torch.zeros(flat_codes.shape[0], grid * grid * bank.num_orientations,
                         device=codes.device, dtype=torch.float32)
    counts.scatter_add_(1, index, torch.ones_like(index, dtype=torch.float32))

    hist = counts.reshape(codes.shape[0], bank.num_scales,
                          grid, grid, bank.num_orientations)
    return hist / hist.norm(dim=-1, keepdim=True).clamp_min(eps)


def magnitude_correlation(clean, adversarial, eps=1e-12):
    """逐图、逐滤波器计算两幅响应图的 Pearson 相关，返回 [N, K]。"""
    if clean.shape != adversarial.shape:
        raise ValueError(f'响应图形状不一致：{tuple(clean.shape)} vs '
                         f'{tuple(adversarial.shape)}')
    a = clean.flatten(2)
    b = adversarial.flatten(2)
    a = a - a.mean(dim=2, keepdim=True)
    b = b - b.mean(dim=2, keepdim=True)
    numerator = (a * b).sum(dim=2)
    denominator = a.norm(dim=2) * b.norm(dim=2)
    return torch.where(denominator > eps, numerator / denominator,
                       torch.full_like(numerator, float('nan')))


def energy_ratio_db(clean, adversarial, eps=1e-12):
    """逐图、逐滤波器计算对抗/干净响应能量的 dB 比，返回 [N, K]。"""
    if clean.shape != adversarial.shape:
        raise ValueError(f'响应图形状不一致：{tuple(clean.shape)} vs '
                         f'{tuple(adversarial.shape)}')
    clean_energy = clean.flatten(2).pow(2).mean(dim=2)
    adv_energy = adversarial.flatten(2).pow(2).mean(dim=2)
    ratio = (adv_energy + eps) / (clean_energy + eps)
    return 10.0 * torch.log10(ratio)


def code_agreement(clean_codes, adversarial_codes):
    """逐图、逐尺度统计方向编码一致率，返回 [N, S]。"""
    if clean_codes.shape != adversarial_codes.shape:
        raise ValueError(f'编码形状不一致：{tuple(clean_codes.shape)} vs '
                         f'{tuple(adversarial_codes.shape)}')
    return (clean_codes == adversarial_codes).to(torch.float32).mean(dim=(-2, -1))


def angular_error_deg(clean_codes, adversarial_codes, bank):
    """逐图、逐尺度统计主方向的圆周平均误差（度），返回 [N, S]。

    方向以 180° 为周期；误差取 ``min(|Δ|, 180-|Δ|)``，范围为 [0, 90]。
    """
    if clean_codes.shape != adversarial_codes.shape:
        raise ValueError(f'编码形状不一致：{tuple(clean_codes.shape)} vs '
                         f'{tuple(adversarial_codes.shape)}')
    angles = bank.scale_angles.to(device=clean_codes.device,
                                  dtype=torch.float32)
    delta = (angles[clean_codes.long()] - angles[adversarial_codes.long()]).abs()
    delta = torch.minimum(delta, 180.0 - delta)
    return delta.mean(dim=(-2, -1))


def code_transition(clean_codes, adversarial_codes, bank):
    """逐图、逐尺度统计方向码转移计数，返回 (trans [N,S,O,O], counts [N,S,O])。

    ``trans[n, s, i, j]`` 是干净码为 i、对抗码为 j 的像素数；``counts`` 是干净码
    为 i 的像素数。用 scatter_add 统计，不需要构造 [N,S,H,W,O] 的 one-hot，
    6000 张图可以按批处理而不占大显存。
    """
    if clean_codes.shape != adversarial_codes.shape:
        raise ValueError(f'编码形状不一致：{tuple(clean_codes.shape)} vs '
                         f'{tuple(adversarial_codes.shape)}')
    if clean_codes.shape[1] != bank.num_scales:
        raise ValueError('编码的尺度数与 bank 不一致')
    orientations = bank.num_orientations
    clean_flat = clean_codes.flatten(2).long()             # [N, S, P]
    adv_flat = adversarial_codes.flatten(2).long()
    index = clean_flat * orientations + adv_flat
    trans = torch.zeros(clean_codes.shape[0], bank.num_scales,
                        orientations * orientations,
                        device=clean_codes.device, dtype=torch.float32)
    trans.scatter_add_(2, index, torch.ones_like(index, dtype=torch.float32))
    counts = torch.zeros(clean_codes.shape[0], bank.num_scales, orientations,
                         device=clean_codes.device, dtype=torch.float32)
    counts.scatter_add_(2, clean_flat, torch.ones_like(clean_flat, dtype=torch.float32))
    return trans.reshape(clean_codes.shape[0], bank.num_scales,
                         orientations, orientations), counts


def code_retention(clean_codes, adversarial_codes, bank):
    """逐图、逐尺度、逐方向统计“干净编码被保留”的条件概率。

    返回 [N, S, O]：第 k 个值 = P(对抗编码 == k | 干净编码 == k)。
    干净样本中从未出现的方向编码返回 NaN，聚合时应使用 nan 忽略。
    """
    trans, counts = code_transition(clean_codes, adversarial_codes, bank)
    retained = trans.diagonal(dim1=-2, dim2=-1) / counts.clamp_min(1.0)
    return torch.where(counts > 0, retained, torch.full_like(retained, float('nan')))
