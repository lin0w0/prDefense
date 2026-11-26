"""深度特征几何分析：干净、对抗与随机噪声掌纹样本。

在项目根目录运行：
    python scripts/analyze_feature_geometry.py
    python scripts/analyze_feature_geometry.py --limit 1200 --balance-classes
    python scripts/analyze_feature_geometry.py --model resnet18
    python scripts/analyze_feature_geometry.py --epsilon 4 --attack pgd --attack cw
    python scripts/analyze_feature_geometry.py --noise-level 1 4
    python scripts/analyze_feature_geometry.py --noise-mode gaussian --noise-level 1
    python scripts/analyze_feature_geometry.py --no-save

流程：
    1. 读取 session2 干净样本、outputs/adv 下同编号的对抗样本，以及在线生成的
       普通随机噪声样本（默认是 L∞ 球内均匀噪声，同样钳制到 [0, 1] 并量化到
       8 位，与对抗样本同口径）；
    2. 读取 session1（训练批次）同身份样本并提取嵌入，作为身份参考：
       每个身份的类别中心（真实身份中心），以及由该身份 10 张训练样本张成的
       局部 PCA 子空间；
    3. 对每个条件计算类内距离、类间距离、到真实身份中心的距离、到最近错误
       身份中心的距离与特征空间最近中心识别率；
    4. 把每张图投影到本身份的训练局部 PCA 子空间，计算重构残差，以及残差
       相对“到身份中心距离”的比值；
    5. 抽样若干身份，用 PCA、t-SNE 画出各条件在嵌入空间的分布，并为若干
       身份单独画局部放大图，图中星号为 session1 训练得到的身份中心。

约定：
    * 嵌入取分类层输入（ResNet-18 为 512 维，VGG-16 为 4096 维），与
      analyze_lpf_similarity.py、analyze_bitplane_noise.py 保持一致；
    * 距离同时按余弦距离和欧氏距离统计，不做特征标准化，保持原嵌入尺度；
    * “真实身份中心”只用 session1 训练样本计算，不参与任何对抗样本生成；
    * 对抗样本按模型、攻击、预算分别统计，CW 使用其 c/kappa 运行目录；
    * 随机噪声默认在 L∞ 球内均匀采样，--noise-level 单位是 1/255、默认 4，
      与最强 L∞ 预算一致；--noise-mode gaussian 可改用高斯噪声；
    * 结果 CSV 的 identity 列为数据集 0 基标签，图中身份编号显示为 identity+1，
      文件名 00001–00010 对应身份 001；
    * 深度特征默认保存到 results/feature_geometry/features/<时间>_session<批次>_<模型>/：
      每个条件一个 npz（features/labels/filenames），另有 train_session<批次>.npz、
      reference.npz（身份中心与局部 PCA 基）和 manifest.json；保存的是分类层输入的
      原始嵌入，未做 L2 归一化，与脚本内计算距离/PCA 的数值完全一致。可加
      --no-save-features 关闭，或用 --feature-dtype float16 减半占用。
"""

import argparse
import inspect
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
from torchvision import transforms

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from data.dataset import (  # noqa: E402
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, PalmprintDataset, make_session_dataset,
)
from defense.features import final_classifier  # noqa: E402
from models.manager import (  # noqa: E402
    DEFAULT_CONFIG, load_config, load_pretrained, resolve_path, select_device,
)
from utils.features import (  # noqa: E402
    extract_embeddings, save_embeddings, save_identity_reference,
)

DEFAULT_ATTACKS = ('pgd', 'bim', 'cw', 'apgd')
ATTACK_COLORS = {
    'pgd': '#D55E00',   # Okabe-Ito 朱红
    'bim': '#0072B2',   # 蓝
    'cw': '#009E73',    # 蓝绿
    'apgd': '#CC79A7',  # 紫红
}
ATTACK_MARKERS = {'pgd': 'o', 'bim': 's', 'cw': '^', 'apgd': 'D'}
CLEAN_COLOR = '#000000'
CLEAN_MARKER = 'o'
NOISE_COLOR = '#7A7A7A'
NOISE_MARKER = 'X'

META_COLUMNS = [
    'model', 'architecture', 'embedding_dim', 'session', 'train_session',
    'condition', 'condition_label', 'run', 'samples', 'classes',
]
DISTANCE_COLUMNS = [
    'intra_cosine_mean', 'intra_cosine_std', 'inter_cosine',
    'to_true_cosine_mean', 'to_true_cosine_std', 'to_true_cosine_median',
    'nearest_other_cosine_mean', 'margin_cosine_mean', 'to_true_cosine_over_inter',
    'nearest_center_acc_pct',
    'intra_euclidean_mean', 'intra_euclidean_std', 'inter_euclidean',
    'to_true_euclidean_mean', 'to_true_euclidean_std', 'to_true_euclidean_median',
    'nearest_other_euclidean_mean', 'margin_euclidean_mean',
    'to_true_euclidean_over_inter',
]
RESIDUAL_COLUMNS = [
    'subspace_rank_mean', 'subspace_variance',
    'residual_mean', 'residual_std', 'residual_median', 'residual_p95',
    'residual_ratio_mean', 'residual_ratio_median', 'distance_to_center_mean',
]
PIXEL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', action='append',
                        help='模型名称，可重复；默认分析配置中的全部模型')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                        help='模型登记与数据集配置文件')
    parser.add_argument('--session', type=int, help='被分析的采集批次，默认 2（验证批次）')
    parser.add_argument('--train-session', type=int, default=1,
                        help='构建身份中心与局部子空间的训练批次，默认 1')
    parser.add_argument('--data-root', type=Path, help='包含 Tongji/ROI 的数据根目录')
    parser.add_argument('--adv-root', type=Path, help='对抗样本根目录，默认读取 attacks.output_dir')
    parser.add_argument('--device', default='auto', help='auto、cpu 或 cuda[:编号]')
    parser.add_argument('--batch-size', type=int, default=64, help='推理批量')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader 工作进程数')
    parser.add_argument('--limit', type=int, help='干净与对抗样本都只取 N 张，便于快速自检')
    parser.add_argument('--balance-classes', action='store_true',
                        help='配合 --limit 按类别均匀抽样，覆盖全部 600 类')
    parser.add_argument('--attack', action='append', choices=DEFAULT_ATTACKS,
                        help='要分析的攻击，可重复；默认全部')
    parser.add_argument('--epsilon', type=float,
                        help='只保留某个 epsN 预算，单位 1/255；默认分析全部预算')
    parser.add_argument('--noise-mode', choices=RandomNoiseDataset.MODES,
                        default='uniform',
                        help='随机噪声类型，默认 uniform（L∞ 球内均匀噪声，与攻击同预算）')
    parser.add_argument('--noise-level', type=float, nargs='+', default=[4.0],
                        help='噪声强度，单位 1/255，可给多个；uniform 为 L∞ 上限，'
                             'gaussian 为标准差；默认 4')
    parser.add_argument('--no-noise', action='store_true', help='不生成随机噪声样本')
    parser.add_argument('--subspace-variance', type=float, default=0.95,
                        help='局部 PCA 子空间保留的累计方差比例，默认 0.95')
    parser.add_argument('--viz-classes', type=int, default=60,
                        help='PCA/t-SNE 图中展示的身份数，默认 60')
    parser.add_argument('--viz-per-class', type=int, default=4,
                        help='每个身份在每个条件下最多取多少张，默认 4')
    parser.add_argument('--zoom-classes', type=int, default=6,
                        help='局部放大图中单独展示的身份数，默认 6')
    parser.add_argument('--tsne-perplexity', type=float, default=30.0,
                        help='t-SNE perplexity，默认 30；会按样本数自动截断')
    parser.add_argument('--seed', type=int, default=42, help='噪声、PCA 与 t-SNE 的随机种子')
    parser.add_argument('--output', type=Path, help='结果输出目录')
    parser.add_argument('--features-dir', type=Path,
                        help='深度特征保存根目录，默认 <output>/features')
    parser.add_argument('--feature-dtype', choices=('float32', 'float16'), default='float32',
                        help='特征保存精度，默认 float32；float16 占用减半但精度略降')
    parser.add_argument('--no-save-features', action='store_true',
                        help='不保存提取的深度特征')
    parser.add_argument('--per-sample', action='store_true', help='额外保存逐图指标 CSV')
    parser.add_argument('--no-save', action='store_true', help='只打印汇总，不画图、不写文件')
    return parser.parse_args()


def balanced_subset(samples, limit):
    """从完整清单中均匀抽取 limit 张，保证每个类别都有样本。

    每个类先分到 limit // 类数 张，余数依次分给前几个类；类内取前若干张。
    limit 小于类别数时无法覆盖全部类别，直接报错。
    """
    by_class = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)
    classes = sorted(by_class)
    base, extra = divmod(limit, len(classes))
    if base < 1:
        raise ValueError(f'limit={limit} 小于类别数 {len(classes)}，无法覆盖全部类别')

    selected = []
    for index, label in enumerate(classes):
        count = min(base + (1 if index < extra else 0), len(by_class[label]))
        selected.extend(by_class[label][:count])
    return selected


def make_pixel_dataset(session=2, root=None, limit=None, balance_classes=False):
    """读取与评估一致的样本清单，输出 [0, 1] 像素空间的三通道图像。"""
    dataset = make_session_dataset(session, root)
    samples = dataset.samples
    if limit is not None:
        if balance_classes:
            samples = balanced_subset(samples, limit)
        else:
            samples = samples[:limit]
    return PalmprintDataset(samples, PIXEL_TRANSFORM)


def make_adv_dataset(clean_dataset, session_dir):
    """按干净样本的文件名，从对抗样本目录读取同编号的 PNG。"""
    session_dir = Path(session_dir)
    samples = []
    for path, label in clean_dataset.samples:
        adv_path = session_dir / f'{path.stem}.png'
        if not adv_path.is_file():
            raise FileNotFoundError(f'{session_dir} 缺少 {adv_path.name}')
        samples.append((adv_path, label))
    return PalmprintDataset(samples, PIXEL_TRANSFORM)


class RandomNoiseDataset(Dataset):
    """在干净样本上叠加普通随机噪声，并量化到 8 位，与对抗样本同口径。

    uniform  每个像素在 [-level, level] 内均匀采样，level 即 L∞ 预算，
             与 PGD/BIM/APGD 的逐像素最大扰动一致，是更公平的对照；
    gaussian 均值 0、标准差 level 的高斯噪声。注意同样 level 下高斯噪声的
             L2 范数远大于 L∞ 攻击（σ=4/255 时约 6.1，而 PGD ε=4/255 约 4.1）。

    噪声按样本序号播种，因此结果与 DataLoader 的批大小、进程数无关；
    灰度图的三个通道使用同一份噪声，并保持三通道相同。
    """

    MODES = ('uniform', 'gaussian')

    def __init__(self, base, level, mode='uniform', seed=0):
        if level < 0:
            raise ValueError('噪声强度不能为负')
        if mode not in self.MODES:
            raise ValueError(f'不支持噪声类型 {mode!r}，可选 {", ".join(self.MODES)}')
        self.base = base
        self.level = float(level)
        self.mode = mode
        self.seed = int(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        generator = np.random.default_rng(self.seed + index)
        shape = (1, *image.shape[-2:])
        if self.mode == 'uniform':
            noise = generator.uniform(-self.level, self.level, size=shape)
        else:
            noise = generator.normal(0.0, self.level, size=shape)
        noise = noise.astype(np.float32)
        noisy = torch.clamp(image[:1] + torch.from_numpy(noise), 0.0, 1.0)
        noisy = torch.round(noisy * 255.0) / 255.0
        return noisy.repeat(3, 1, 1), label


def epsilon_of(name):
    """从 eps1 / eps4 这样的目录名解析 epsilon，其他名称返回 None。"""
    match = re.fullmatch(r'eps(\d+(?:\.\d+)?)', name)
    return float(match.group(1)) if match else None


def find_attack_runs(adv_root, model_name, attack, session, epsilon=None):
    """定位一次攻击的全部运行目录，返回 [(运行目录, session 子目录), ...]。"""
    attack_dir = Path(adv_root) / attack / model_name
    if not attack_dir.is_dir():
        return []
    runs = sorted(directory for directory in attack_dir.iterdir()
                  if directory.is_dir() and (directory / f'session{session}').is_dir())
    if not runs:
        return []

    eps_runs = [directory for directory in runs if epsilon_of(directory.name) is not None]
    if eps_runs:
        if epsilon is not None:
            eps_runs = [directory for directory in eps_runs
                        if epsilon_of(directory.name) == float(epsilon)]
        eps_runs.sort(key=lambda directory: epsilon_of(directory.name))
        return [(run, run / f'session{session}') for run in eps_runs]

    if epsilon is not None:
        return []
    if len(runs) > 1:
        names = '、'.join(directory.name for directory in runs)
        raise ValueError(f'{attack_dir} 有多个非 epsilon 运行（{names}），请手动清理')
    return [(runs[0], runs[0] / f'session{session}')]


def attack_label(attack, run_dir, session_dir):
    """图例名称：L∞ 攻击标注 epsilon，CW 标注 c。"""
    eps = epsilon_of(run_dir.name)
    if eps is not None:
        return f'{attack.upper()}（ε={eps:g}/255）'
    try:
        with (Path(session_dir) / 'run_info.json').open(encoding='utf-8') as file:
            params = json.load(file).get('params', {})
    except (OSError, ValueError):
        params = {}
    if attack == 'cw' and 'c' in params:
        return f"CW（c={params['c']:g}）"
    return attack.upper()


def build_identity_reference(features, labels, num_classes, variance):
    """用训练集同身份样本构建身份中心与局部 PCA 子空间。

    返回：
        centers       [num_classes, D] 每个身份的训练特征均值；
        bases         每个身份的 PCA 基 [r, D]，r 由累计方差比例决定；
        present       [num_classes] 标记训练集中是否出现过该身份；
        ranks         每个出现身份的子空间维数。
    """
    dim = features.shape[1]
    centers = torch.zeros(num_classes, dim)
    present = torch.zeros(num_classes, dtype=torch.bool)
    bases = [features.new_zeros((0, dim)) for _ in range(num_classes)]
    ranks = []

    for label in range(num_classes):
        index = (labels == label).nonzero(as_tuple=True)[0]
        if index.numel() == 0:
            continue
        samples = features[index]
        mean = samples.mean(dim=0)
        centers[label] = mean
        present[label] = True

        centered = samples - mean
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        energy = singular.square()
        total = energy.sum()
        if total <= 0:
            rank = 0
        else:
            cumulative = torch.cumsum(energy, dim=0) / total
            rank = int((cumulative < variance).sum().item()) + 1
            # 同身份 n 个训练样本的中心化数据秩最多为 n-1。
            rank = min(rank, vh.shape[0], max(centered.shape[0] - 1, 1))
        bases[label] = vh[:rank].contiguous()
        ranks.append(rank)

    return centers, bases, present, ranks


def mean_pairwise_distance(points, metric):
    """返回点集内两两距离的均值；不足两个点时返回 nan。"""
    if points.shape[0] < 2:
        return float('nan')
    if metric == 'euclidean':
        distances = torch.pdist(points)
    else:
        unit = F.normalize(points, dim=1)
        similarity = unit @ unit.T
        upper = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
        distances = 1.0 - similarity[upper]
    return distances.mean().item()


def distances_to_centers(features, centers, metric):
    """返回 [N, num_classes] 的逐样本到各类中心的距离。"""
    if metric == 'euclidean':
        return torch.cdist(features, centers)
    unit = F.normalize(features, dim=1)
    unit_centers = F.normalize(centers, dim=1)
    return 1.0 - unit @ unit_centers.T


def per_class_intra(features, labels, metric):
    """类内距离：先算每个身份内部两两距离的均值，再对身份等权平均。"""
    means = []
    for label in labels.unique():
        index = (labels == label).nonzero(as_tuple=True)[0]
        if index.numel() < 2:
            continue
        means.append(mean_pairwise_distance(features[index], metric))
    if not means:
        return float('nan'), float('nan')
    values = torch.tensor(means)
    return values.mean().item(), values.std(unbiased=False).item()


def condition_centers(features, labels):
    """返回当前条件下每个身份的特征均值，用于计算类间距离。"""
    return torch.stack([features[labels == label].mean(dim=0)
                        for label in labels.unique()])


def summarize_distances(features, labels, centers, present, scales):
    """计算一个条件的类内、类间与身份中心距离，返回汇总和逐样本数组。"""
    summary = {
        'samples': int(features.shape[0]),
        'classes': int(labels.unique().numel()),
    }
    per_sample = {}
    centers_this_condition = condition_centers(features, labels)

    for metric in ('cosine', 'euclidean'):
        intra_mean, intra_std = per_class_intra(features, labels, metric)
        # 类间距离使用该条件自身的身份中心，而不是固定的训练中心，
        # 这样能看出对抗特征是否仍按身份聚成 600 个可分的簇。
        inter_mean = mean_pairwise_distance(centers_this_condition, metric)

        distances = distances_to_centers(features, centers, metric)
        if not bool(present.all()):
            distances[:, ~present] = float('inf')
        own = distances.gather(1, labels[:, None]).squeeze(1)
        # 先在包含真实中心的距离里取最近中心，用于特征空间身份识别率；
        # 再把真实中心遮掉，得到“最近错误身份中心”。
        nearest_label = distances.argmin(dim=1)
        distances.scatter_(1, labels[:, None], float('inf'))
        nearest_other = distances.min(dim=1).values

        summary[f'intra_{metric}_mean'] = intra_mean
        summary[f'intra_{metric}_std'] = intra_std
        summary[f'inter_{metric}'] = inter_mean
        summary[f'to_true_{metric}_mean'] = own.mean().item()
        summary[f'to_true_{metric}_std'] = own.std(unbiased=False).item()
        summary[f'to_true_{metric}_median'] = own.median().item()
        summary[f'nearest_other_{metric}_mean'] = nearest_other.mean().item()
        summary[f'margin_{metric}_mean'] = (nearest_other - own).mean().item()
        summary[f'to_true_{metric}_over_inter'] = (own.mean().item() / scales[metric]
                                                   if scales[metric] else float('nan'))
        if metric == 'cosine':
            summary['nearest_center_acc_pct'] = (
                100.0 * (nearest_label == labels).float().mean().item())

        per_sample[f'to_true_{metric}'] = own.numpy()
        per_sample[f'nearest_other_{metric}'] = nearest_other.numpy()
        del distances

    return summary, per_sample


def summarize_residuals(features, labels, centers, bases):
    """计算每个样本到本身份局部 PCA 子空间的重构残差。

    residual          ||(x - μ_c) - V_c^T V_c (x - μ_c)||，与嵌入同尺度；
    residual_ratio    residual / ||x - μ_c||，即偏离身份中心的方向落在子空间
                      之外的比例，取值约在 [0, 1]，便于跨条件比较；
    distance_to_center||x - μ_c||，即样本到真实身份中心的欧氏距离。
    """
    total = features.shape[0]
    residual = torch.full((total,), float('nan'))
    ratio = torch.full((total,), float('nan'))
    distance = torch.full((total,), float('nan'))

    for label in labels.unique():
        index = (labels == label).nonzero(as_tuple=True)[0]
        delta = features[index] - centers[label]
        norm = delta.norm(dim=1)
        basis = bases[int(label)]
        if basis.shape[0] == 0:
            residual_values = norm.clone()
        else:
            projection = (delta @ basis.T) @ basis
            residual_values = (delta - projection).norm(dim=1)

        valid = norm > 1e-12
        residual[index] = residual_values
        distance[index] = norm
        ratio[index] = torch.where(valid, residual_values / norm.clamp_min(1e-12),
                                   torch.zeros_like(residual_values))

    summary = {
        'residual_mean': residual.mean().item(),
        'residual_std': residual.std(unbiased=False).item(),
        'residual_median': residual.median().item(),
        'residual_p95': residual.quantile(0.95).item(),
        'residual_ratio_mean': ratio.mean().item(),
        'residual_ratio_median': ratio.median().item(),
        'distance_to_center_mean': distance.mean().item(),
    }
    arrays = {
        'residual': residual.numpy(),
        'residual_ratio': ratio.numpy(),
        'distance_to_center': distance.numpy(),
    }
    return summary, arrays


def analyze_condition(features, labels, centers, bases, present, scales):
    """汇总一个条件的全部距离与重构残差指标。"""
    summary, per_sample = summarize_distances(features, labels, centers, present, scales)
    residual_summary, residual_arrays = summarize_residuals(
        features, labels, centers, bases)
    summary.update(residual_summary)
    per_sample.update(residual_arrays)
    return summary, per_sample


def pick_classes(present_classes, count):
    """从实际出现的身份中均匀抽取 count 个，保证图的可读性和可复现性。"""
    classes = [int(label) for label in present_classes]
    if count >= len(classes):
        return classes
    positions = np.linspace(0, len(classes) - 1, count).round().astype(int)
    return [classes[position] for position in np.unique(positions)]


def select_visual_mask(labels, classes, per_class):
    """选出用于可视化的行：每个身份取前 per_class 张，所有条件共用同一掩码。"""
    mask = torch.zeros(len(labels), dtype=torch.bool)
    for label in classes:
        index = (labels == label).nonzero(as_tuple=True)[0]
        if index.numel():
            mask[index[:per_class]] = True
    return mask


def fit_embedding(pooled, seed):
    """在合并后的特征上拟合 PCA，返回最多 50 维坐标与 PCA 对象。"""
    samples, dim = pooled.shape
    if samples < 4:
        raise ValueError(f'可视化样本太少（{samples} 张），请增大 --viz-classes 或 --viz-per-class')
    components = max(2, min(50, samples - 1, dim))
    solver = 'randomized' if components < min(samples, dim) else 'full'
    pca = PCA(n_components=components, svd_solver=solver, random_state=seed)
    return pca.fit_transform(pooled), pca


def fit_tsne(coordinates, perplexity, seed):
    """在 PCA 前若干维上运行 t-SNE；自动处理 sklearn 版本的迭代参数名。"""
    samples = coordinates.shape[0]
    perplexity = float(min(perplexity, max(2.0, (samples - 1) / 3.0)))
    parameters = {
        'n_components': 2,
        'perplexity': perplexity,
        'init': 'pca',
        'learning_rate': 'auto',
        'random_state': seed,
    }
    if 'max_iter' in inspect.signature(TSNE).parameters:
        parameters['max_iter'] = 1000
    else:
        parameters['n_iter'] = 1000
    return TSNE(**parameters).fit_transform(coordinates)


def apply_plot_style():
    """Windows 自带字体可以正常显示中文坐标轴和标题。"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'


def scatter_entry(axis, coordinates, entry, hollow_size_bonus=6):
    """按条件画一层散点：实心表示较强预算，空心表示较弱预算。"""
    if entry['filled']:
        axis.scatter(coordinates[:, 0], coordinates[:, 1], s=entry['size'],
                     marker=entry['marker'], color=entry['color'],
                     alpha=entry['alpha'], linewidths=0, zorder=entry['zorder'])
    else:
        axis.scatter(coordinates[:, 0], coordinates[:, 1],
                     s=entry['size'] + hollow_size_bonus, marker=entry['marker'],
                     facecolors='none', edgecolors=entry['color'], alpha=0.9,
                     linewidths=0.9, zorder=entry['zorder'])


def legend_handles(entries):
    """按条件生成图例句柄，空心/实心与散点一致。"""
    handles = []
    for entry in entries:
        handles.append(Line2D(
            [], [], linestyle='none', marker=entry['marker'], markersize=6.5,
            markerfacecolor=entry['color'] if entry['filled'] else 'none',
            markeredgecolor=entry['color'], markeredgewidth=1.2,
            color='none', label=entry['label']))
    return handles


def draw_overview(model_name, entries, pca_2d, tsne_2d, explained, prefix):
    """一个模型一张图：左 PCA、右 t-SNE，颜色区分样本类型。"""
    apply_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(13.6, 6.4))
    panels = [
        (axes[0], pca_2d,
         f'PCA（PC1 {explained[0] * 100:.1f}%，PC2 {explained[1] * 100:.1f}%）',
         'PC1', 'PC2'),
        (axes[1], tsne_2d, 't-SNE（在 PCA 降维结果上拟合）', 't-SNE 1', 't-SNE 2'),
    ]
    for axis, coordinates, title, xlabel, ylabel in panels:
        for entry in entries:
            scatter_entry(axis, coordinates[entry['slice']], entry)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=8)

    figure.legend(legend_handles(entries), [entry['label'] for entry in entries],
                  loc='lower center', ncol=4, frameon=False, fontsize=9,
                  title='样本类型（空心 = 较弱预算或噪声强度）')
    figure.suptitle(f'{model_name}：干净、对抗与随机噪声样本的深度特征分布', fontsize=13)
    figure.tight_layout(rect=(0, 0.11, 1, 0.95))

    png_path = prefix.with_name(prefix.name + '_pca_tsne.png')
    pdf_path = prefix.with_name(prefix.name + '_pca_tsne.pdf')
    figure.savefig(png_path, dpi=220, bbox_inches='tight')
    figure.savefig(pdf_path, bbox_inches='tight')
    plt.close(figure)
    return png_path, pdf_path


def draw_class_zoom(model_name, entries, zoom_classes, labels_pooled,
                    pca_2d, pca, centers, prefix):
    """每个身份一个小图：干净样本、各条件与训练身份中心在 PCA 空间中的位置。"""
    apply_plot_style()
    columns = 3
    rows = math.ceil(len(zoom_classes) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 4.0 * rows),
                                squeeze=False)
    center_coordinates = (centers.numpy() - pca.mean_) @ pca.components_[:2].T

    for axis, label in zip(axes.flat, zoom_classes):
        for entry in entries:
            # 每个条件都按同一份可视化掩码取材，因此各切片内的标签顺序
            # 与 labels_pooled 完全一致；不能用全局切片去索引单条件标签。
            expected = entry['slice'].stop - entry['slice'].start
            if expected != len(labels_pooled):
                raise RuntimeError('可视化切片长度与标签数量不一致')
            mask = labels_pooled == label
            scatter_entry(axis, pca_2d[entry['slice']][mask], entry)
        center = center_coordinates[label]
        axis.scatter([center[0]], [center[1]], marker='*', s=170, color='white',
                     edgecolors='black', linewidths=1.2, zorder=10)
        axis.set_title(f'身份 {label + 1:03d}', fontsize=10)
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=7)

    for axis in axes.flat[len(zoom_classes):]:
        axis.set_visible(False)

    figure.legend(legend_handles(entries), [entry['label'] for entry in entries],
                  loc='lower center', ncol=4, frameon=False, fontsize=9,
                  title='样本类型（空心 = 较弱预算或噪声强度；白色星号 = session1 身份中心）')
    figure.suptitle(
        f'{model_name}：各身份局部 PCA 视图（图中身份编号 = 数据集标签 + 1）',
        fontsize=13)
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))

    png_path = prefix.with_name(prefix.name + '_class_zoom.png')
    pdf_path = prefix.with_name(prefix.name + '_class_zoom.pdf')
    figure.savefig(png_path, dpi=220, bbox_inches='tight')
    figure.savefig(pdf_path, bbox_inches='tight')
    plt.close(figure)
    return png_path, pdf_path


def print_distance_table(model_name, entries):
    """按余弦距离打印每个条件的距离指标，便于终端快速比较。"""
    print(f'\n[{model_name}] 距离指标（余弦距离；到真实中心使用 session1 身份中心）')
    header = (f"{'条件':<12}{'样本':>6}{'类内':>9}{'类间':>9}"
              f"{'到真实中心':>11}{'最近错误中心':>13}{'边际':>9}{'识别率':>8}")
    print(header)
    print('-' * len(header))
    for entry in entries:
        summary = entry['summary']
        print(f"{entry['short']:<12}{summary['samples']:>6}"
              f"{summary['intra_cosine_mean']:>9.4f}"
              f"{summary['inter_cosine']:>9.4f}"
              f"{summary['to_true_cosine_mean']:>11.4f}"
              f"{summary['nearest_other_cosine_mean']:>13.4f}"
              f"{summary['margin_cosine_mean']:>9.4f}"
              f"{summary['nearest_center_acc_pct']:>7.1f}%")
    print('类内/类间越小表示同类更紧、不同类中心更近；边际 = 最近错误中心 - 真实中心，越大越安全。')


def print_residual_table(model_name, entries, rank_mean):
    """打印每个条件到局部 PCA 子空间的重构残差。"""
    print(f'\n[{model_name}] 局部 PCA 重构残差（训练子空间平均秩 {rank_mean:.1f}）')
    header = (f"{'条件':<12}{'残差均值':>12}{'残差中位数':>13}"
              f"{'残差/中心距':>12}{'到中心距离':>13}{'残差P95':>12}")
    print(header)
    print('-' * len(header))
    for entry in entries:
        summary = entry['summary']
        print(f"{entry['short']:<12}{summary['residual_mean']:>12.4f}"
              f"{summary['residual_median']:>13.4f}"
              f"{summary['residual_ratio_mean']:>12.4f}"
              f"{summary['distance_to_center_mean']:>13.4f}"
              f"{summary['residual_p95']:>12.4f}")
    print('残差/中心距 = 偏离身份中心的方向落在训练子空间外的比例，越接近 1 越异常。')


def main():
    args = parse_args()
    config = load_config(args.config)
    settings = config['evaluation']
    attack_settings = config.get('attacks', {})

    session = args.session if args.session is not None else settings.get('session', 2)
    train_session = args.train_session
    if session not in (1, 2) or train_session not in (1, 2):
        raise SystemExit('采集批次必须为 1 或 2')
    if args.subspace_variance <= 0 or args.subspace_variance > 1:
        raise SystemExit('--subspace-variance 必须在 (0, 1] 内')
    if args.viz_classes < 1 or args.viz_per_class < 1 or args.zoom_classes < 1:
        raise SystemExit('--viz-classes、--viz-per-class、--zoom-classes 必须为正整数')

    data_root = args.data_root or resolve_path(settings.get('data_root', 'data/resources'))
    adv_root = args.adv_root or resolve_path(attack_settings.get('output_dir', 'outputs/adv'))
    output_dir = args.output or resolve_path('results/feature_geometry')
    features_root = args.features_dir or (output_dir / 'features')
    save_features = not args.no_save and not args.no_save_features
    device = select_device(args.device)
    names = args.model or list(config['models'])
    unknown = [name for name in names if name not in config['models']]
    if unknown:
        raise SystemExit(f'配置中没有模型 {", ".join(unknown)}；'
                         f'可选 {", ".join(config["models"])}')

    attacks = args.attack or list(DEFAULT_ATTACKS)
    noise_levels = [] if args.no_noise else sorted({float(v) for v in args.noise_level})
    if any(level < 0 for level in noise_levels):
        raise SystemExit('--noise-level 不能为负')

    print(f'设备：{device}，被分析批次：session{session}，'
          f'训练参考：session{train_session}，数据：{data_root}')
    clean_dataset = make_pixel_dataset(session, data_root, limit=args.limit,
                                       balance_classes=args.balance_classes)
    train_dataset = make_pixel_dataset(train_session, data_root)
    all_labels = torch.tensor([label for _, label in clean_dataset.samples])
    present_labels = torch.unique(all_labels)
    print(f'被分析样本：{len(clean_dataset)} 张，覆盖 {present_labels.numel()} 个身份；'
          f'训练参考：{len(train_dataset)} 张')

    # 所有条件按同一行顺序排列（干净 / 对抗 / 噪声），因此共用一份可视化掩码。
    viz_classes = pick_classes(present_labels, args.viz_classes)
    zoom_classes = pick_classes(viz_classes, args.zoom_classes)
    viz_mask = select_visual_mask(all_labels, viz_classes, args.viz_per_class)
    labels_pooled = all_labels[viz_mask].numpy()
    if not args.no_save:
        print(f'可视化：{len(viz_classes)} 个身份 × 每类最多 {args.viz_per_class} 张 × '
              f'{1 + len(attacks) * 2 + len(noise_levels)} 种条件（对抗预算视运行目录而定）')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    distance_frames, residual_frames, per_sample_frames = [], [], []

    for name in names:
        model, info = load_pretrained(name, device=device, config=config)
        embedding_dim = final_classifier(model).in_features
        print(f"\n[{name}] {info['architecture']}，{embedding_dim} 维嵌入，"
              f"权重 {info['weights']}")

        train_features, train_labels = extract_embeddings(
            model, train_dataset, device, batch_size=args.batch_size,
            num_workers=args.num_workers, desc=f'{name}-session{train_session}')
        centers, bases, present, ranks = build_identity_reference(
            train_features, train_labels, info['num_classes'], args.subspace_variance)
        scales = {
            'cosine': mean_pairwise_distance(centers[present], 'cosine'),
            'euclidean': mean_pairwise_distance(centers[present], 'euclidean'),
        }
        rank_mean = float(np.mean(ranks)) if ranks else float('nan')
        print(f"  session{train_session} 参考：{int(present.sum())} 个身份，"
              f"训练类间距离（余弦）{scales['cosine']:.4f}，"
              f"局部子空间平均秩 {rank_mean:.1f}")
        if not bool(present.all()):
            missing = (~present).nonzero(as_tuple=True)[0].tolist()
            print(f'  警告：session{train_session} 缺少 {len(missing)} 个身份，'
                  f'这些身份的距离指标会是 inf：{missing[:5]}...')

        model_feature_dir = features_root / f'{stamp}_session{session}_{name}'
        if save_features:
            # 训练集特征单独保存，后续可以直接重建身份中心和局部子空间。
            save_embeddings(
                model_feature_dir / f'train_session{train_session}.npz',
                train_features, train_labels,
                [path.name for path, _ in train_dataset.samples],
                args.feature_dtype)
            save_identity_reference(
                model_feature_dir / 'reference.npz', centers, bases, present,
                args.subspace_variance, rank_mean)
        del train_features

        # 组装条件清单；所有数据集与 clean_dataset 保持相同的样本顺序和标签。
        entries = [{
            'condition': 'clean',
            'short': '干净',
            'label': '干净样本',
            'run': 'clean',
            'color': CLEAN_COLOR,
            'marker': CLEAN_MARKER,
            'filled': True,
            'alpha': 0.85,
            'size': 16,
            'zorder': 5,
            'dataset': clean_dataset,
        }]
        for attack in attacks:
            found = find_attack_runs(adv_root, name, attack, session, args.epsilon)
            if not found:
                detail = f'eps{args.epsilon:g}' if args.epsilon is not None else '可用'
                print(f'  跳过 {attack}：{adv_root / attack / name} 没有 {detail} 的 '
                      f'session{session} 运行')
                continue
            eps_values = sorted(epsilon_of(run.name) for run, _ in found
                                if epsilon_of(run.name) is not None)
            for run_dir, session_dir in found:
                epsilon = epsilon_of(run_dir.name)
                # 同一攻击下，较低预算画空心、最高预算画实心，颜色和标记保持一致。
                if epsilon is None:
                    filled, short = True, attack.upper()
                else:
                    filled = True if len(eps_values) <= 1 else epsilon == eps_values[-1]
                    short = f'{attack.upper()} ε{epsilon:g}'
                entries.append({
                    'condition': f'{attack}-{run_dir.name}',
                    'short': short,
                    'label': attack_label(attack, run_dir, session_dir),
                    'run': str(run_dir),
                    'color': ATTACK_COLORS[attack],
                    'marker': ATTACK_MARKERS[attack],
                    'filled': filled,
                    'alpha': 0.72,
                    'size': 12,
                    'zorder': 3,
                    'dataset': make_adv_dataset(clean_dataset, session_dir),
                })
        for rank, level in enumerate(noise_levels):
            # 多个噪声强度时，强度最大的实心，其余空心。
            filled = rank == len(noise_levels) - 1
            symbol = 'ε' if args.noise_mode == 'uniform' else 'σ'
            kind = '均匀' if args.noise_mode == 'uniform' else '高斯'
            entries.append({
                'condition': f'noise-{args.noise_mode}-{level:g}',
                'short': f'噪声{symbol}{level:g}',
                'label': f'{kind}随机噪声（{symbol}={level:g}/255）',
                'run': 'generated',
                'color': NOISE_COLOR,
                'marker': NOISE_MARKER,
                'filled': filled,
                'alpha': 0.72,
                'size': 13,
                'zorder': 4,
                'dataset': RandomNoiseDataset(
                    clean_dataset, level / 255.0, mode=args.noise_mode, seed=args.seed),
            })

        pooled_parts = []
        pool_offset = 0
        filenames = [path.name for path, _ in clean_dataset.samples]
        condition_manifest = []
        for entry in entries:
            features, labels = extract_embeddings(
                model, entry['dataset'], device, batch_size=args.batch_size,
                num_workers=args.num_workers, desc=f"{name}-{entry['short']}")
            if not torch.equal(labels, all_labels):
                raise RuntimeError(f"{entry['condition']} 的样本顺序与干净样本不一致")

            summary, per_sample = analyze_condition(
                features, labels, centers, bases, present, scales)
            summary['subspace_rank_mean'] = rank_mean
            summary['subspace_variance'] = args.subspace_variance
            entry['summary'] = summary

            meta = {
                'model': name,
                'architecture': info['architecture'],
                'embedding_dim': embedding_dim,
                'session': session,
                'train_session': train_session,
                'condition': entry['condition'],
                'condition_label': entry['label'],
                'run': entry['run'],
                'samples': summary['samples'],
                'classes': summary['classes'],
            }
            distance_rows = []
            for key in DISTANCE_COLUMNS:
                distance_rows.append((key, summary.get(key, float('nan'))))
            distance_frames.append({**meta, **dict(distance_rows)})

            residual_row = dict(meta)
            for key in RESIDUAL_COLUMNS:
                residual_row[key] = summary.get(key, float('nan'))
            residual_frames.append(residual_row)

            if args.per_sample and not args.no_save:
                frame = pd.DataFrame({
                    'model': name,
                    'architecture': info['architecture'],
                    'condition': entry['condition'],
                    'condition_label': entry['label'],
                    'run': entry['run'],
                    'filename': filenames,
                    'identity': labels.numpy(),
                    **per_sample,
                })
                per_sample_frames.append(frame)

            condition_file = f"{entry['condition']}.npz"
            if save_features:
                save_embeddings(model_feature_dir / condition_file, features,
                                     labels, filenames, args.feature_dtype)
                condition_manifest.append({
                    'condition': entry['condition'],
                    'condition_label': entry['label'],
                    'run': entry['run'],
                    'samples': int(summary['samples']),
                    'file': condition_file,
                })

            if not args.no_save:
                part = features[viz_mask].numpy().astype(np.float32)
                entry['slice'] = slice(pool_offset, pool_offset + len(part))
                pool_offset += len(part)
                pooled_parts.append(part)

            del features

        print_distance_table(name, entries)
        print_residual_table(name, entries, rank_mean)

        if not args.no_save:
            pooled = np.concatenate(pooled_parts, axis=0)
            coordinates, pca = fit_embedding(pooled, args.seed)
            explained = pca.explained_variance_ratio_
            pca_2d = coordinates[:, :2]
            print(f'  拟合 t-SNE：{pooled.shape[0]} 个样本，'
                  f'{pooled.shape[1]} 维特征 → {coordinates.shape[1]} 维 PCA')
            tsne_2d = fit_tsne(coordinates, args.tsne_perplexity, args.seed)

            prefix = output_dir / f'session{session}_{stamp}_{name}'
            # 先把缩放后的中心投影到 PCA 平面，局部放大图会用到。
            png_path, _ = draw_overview(
                name, entries, pca_2d, tsne_2d, explained, prefix)
            zoom_png, _ = draw_class_zoom(
                name, entries, zoom_classes, labels_pooled, pca_2d, pca, centers, prefix)
            print(f'  已保存：{png_path}')
            print(f'  已保存：{zoom_png}')

        if save_features:
            manifest = {
                'model': name,
                'architecture': info['architecture'],
                'embedding_dim': embedding_dim,
                'num_classes': info['num_classes'],
                'weights': info['weights'],
                'session': session,
                'train_session': train_session,
                'data_root': str(data_root),
                'adv_root': str(adv_root),
                'feature_dtype': args.feature_dtype,
                'preprocessing': {
                    'image_size': IMAGE_SIZE,
                    'mean': IMAGENET_MEAN,
                    'std': IMAGENET_STD,
                    'grayscale_to_three_channels': True,
                },
                'train': {
                    'file': f'train_session{train_session}.npz',
                    'samples': len(train_dataset),
                },
                'reference': {
                    'file': 'reference.npz',
                    'subspace_variance': args.subspace_variance,
                    'mean_rank': rank_mean,
                },
                'conditions': condition_manifest,
                'args': {key: (str(value) if isinstance(value, Path) else value)
                         for key, value in vars(args).items()},
                'created_at': datetime.now().isoformat(timespec='seconds'),
            }
            manifest_path = model_feature_dir / 'manifest.json'
            with manifest_path.open('w', encoding='utf-8') as file:
                json.dump(manifest, file, ensure_ascii=False, indent=2)
            print(f'  特征已保存：{model_feature_dir}（逐条件 npz + manifest.json）')

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if args.no_save:
        return

    distance_table = pd.DataFrame(distance_frames)
    distance_path = output_dir / f'distances_session{session}_{stamp}.csv'
    distance_table.to_csv(distance_path, index=False, encoding='utf-8-sig')
    print(f'\n距离指标 CSV：{distance_path}')

    residual_table = pd.DataFrame(residual_frames)
    residual_path = output_dir / f'residuals_session{session}_{stamp}.csv'
    residual_table.to_csv(residual_path, index=False, encoding='utf-8-sig')
    print(f'重构残差 CSV：{residual_path}')

    if args.per_sample and per_sample_frames:
        per_sample_table = pd.concat(per_sample_frames, ignore_index=True)
        per_sample_path = output_dir / f'per_sample_session{session}_{stamp}.csv'
        per_sample_table.to_csv(per_sample_path, index=False, encoding='utf-8-sig')
        print(f'逐图指标 CSV：{per_sample_path}')


if __name__ == '__main__':
    main()
