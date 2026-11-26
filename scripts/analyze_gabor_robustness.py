"""多尺度、多方向 Gabor 方向响应与抗扰动分析。

在项目根目录运行：
    python scripts/analyze_gabor_robustness.py
    python scripts/analyze_gabor_robustness.py --limit 1200 --balance-classes
    python scripts/analyze_gabor_robustness.py --dominance min        # CompCode 式最小响应规则
    python scripts/analyze_gabor_robustness.py --model resnet18 --epsilon 4
    python scripts/analyze_gabor_robustness.py --wavelengths 4 8 16 24 --orientations 8
    python scripts/analyze_gabor_robustness.py --no-identity --no-save

流程：
    1. 读取与评估相同文件编号的干净图像，以及 outputs/adv 下各攻击、各预算的
       对抗样本；只做图像统计，不做模型前向，因此不需要加载模型权重；
    2. 用多尺度 Gabor 滤波器组（默认 λ=4/6/8/12/16/24，6 个方向）提取每个
       尺度、每个方向的解析响应幅度；
    3. 对每个尺度按竞争规则得到方向码（--dominance max 取响应最强，
       min 为 CompCode 式最小响应规则）；
    4. 比较干净/对抗响应的相关性、能量比、方向码一致率、码保留率与方向
       转移矩阵，按尺度、方向汇总；
    5. 用分块方向码直方图作为身份描述子，在另一个 session 的干净图库上做
       Rank-1/Rank-5 检索，并统计同图干净/对抗描述子余弦（身份锚点稳定性）。

输出（results/gabor_robustness/）：
    session2_<规则>_<时间>_robustness.csv   一行一个「条件 × 尺度 × 方向」
    session2_<规则>_<时间>_scale.csv        一行一个「条件 × 尺度」
    session2_<规则>_<时间>_identity.csv     一行一个「条件 × 特征集合」
    session2_<规则>_<时间>_transitions.csv  方向码转移矩阵
    session2_<规则>_<时间>_images.csv       逐图指标（仅 --per-image）
    *_retention.png / *_corr.png             尺度×方向的码保留率/响应相关性热图
    *_curves.png                             尺度、方向、响应相关性的曲线
    *_identity.png                           Rank-1 与身份锚点余弦
    *_transition.png                         方向码转移矩阵

指标口径：
    * magnitude_corr：同一图、同一滤波器下，干净与对抗响应图的逐像素 Pearson
      相关，再对图像取均值；1 表示局部方向响应完全不变。
    * energy_ratio_db：对抗/干净响应能量的 dB 比；大于 0 表示该尺度/方向被
      注入了额外能量（高频对抗扰动通常如此）。
    * code_agreement：同一图、同一尺度下，干净与对抗方向码相同的像素比例。
    * code_retention：条件保留率 P(对抗码 = k | 干净码 = k)，按方向 k 统计；
      mean/std 是逐图等权平均，pooled 是按像素数加权的总体比例。
    * identity：方向码分块直方图的 Rank-1/Rank-5，查询为当前条件、图库为
      另一 session 的干净图；anchor_cosine 是同一图干净/对抗描述子的余弦。
"""

import argparse
import json
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
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from data.dataset import IMAGE_SIZE, PalmprintDataset, make_session_dataset  # noqa: E402
from defense.gabor import (  # noqa: E402
    DEFAULT_ORIENTATIONS, DEFAULT_WAVELENGTHS, DOMINANCE_RULES,
    angular_error_deg, code_agreement, code_transition, dominant_orientation,
    energy_ratio_db, gabor_magnitude, magnitude_correlation, make_gabor_bank,
    orientation_histogram,
)
from models.manager import (  # noqa: E402
    DEFAULT_CONFIG, load_config, resolve_path, select_device,
)

DEFAULT_ATTACKS = ('pgd', 'bim', 'cw', 'apgd')
CLEAN_COLOR = '#333333'
ATTACK_COLORS = {
    'pgd': '#D55E00',
    'bim': '#0072B2',
    'cw': '#009E73',
    'apgd': '#CC79A7',
}
ATTACK_MARKERS = {'pgd': 'o', 'bim': 's', 'cw': '^', 'apgd': 'D'}
PIXEL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),          # 灰度图 -> [1, H, W]，取值 [0, 1]
])


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', action='append',
                        help='模型名称，可重复；默认分析配置中的全部模型')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='模型登记与数据集配置文件')
    parser.add_argument('--session', type=int, help='同济采集批次，1 或 2')
    parser.add_argument('--data-root', type=Path, help='包含 Tongji/ROI 的数据根目录')
    parser.add_argument('--adv-root', type=Path, help='对抗样本根目录，默认读取 attacks.output_dir')
    parser.add_argument('--attack', action='append', choices=DEFAULT_ATTACKS,
                        help='要分析的攻击，可重复；默认全部')
    parser.add_argument('--epsilon', type=float,
                        help='只保留某个 epsN 预算，单位 1/255；默认分析全部预算')
    parser.add_argument('--limit', type=int, help='干净与对抗样本都只取 N 张，便于快速自检')
    parser.add_argument('--balance-classes', action='store_true',
                        help='配合 --limit 按类别均匀抽样，覆盖全部 600 类')

    parser.add_argument('--wavelengths', type=float, nargs='+',
                        default=list(DEFAULT_WAVELENGTHS),
                        help='Gabor 载波波长 λ（像素），默认 4 6 8 12 16 24')
    parser.add_argument('--orientations', type=int, default=DEFAULT_ORIENTATIONS,
                        help='等分 180° 的方向数，默认 6')
    parser.add_argument('--angles', type=float, nargs='+',
                        help='显式方向角度（度），设置后覆盖 --orientations')
    parser.add_argument('--sigma-ratio', type=float, default=0.56,
                        help='空间包络标准差 / λ，默认 0.56（约一个倍频程）')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='包络长宽比，默认 0.5；越小沿纹线方向拉得越长')
    parser.add_argument('--pad-ratio', type=float, default=3.0,
                        help='反射填充宽度 = pad_ratio × 最大包络标准差，默认 3')
    parser.add_argument('--dominance', default='max', choices=DOMINANCE_RULES,
                        help='方向码竞争规则：max 取最强响应（默认），min 为 CompCode 式最小响应')
    parser.add_argument('--code-grid', type=int, default=4,
                        help='身份描述子的分块网格 G×G，默认 4')

    parser.add_argument('--device', default='auto', help='auto、cpu 或 cuda[:编号]')
    parser.add_argument('--batch-size', type=int, default=16, help='图像批量')
    parser.add_argument('--filter-chunk', type=int, default=8,
                        help='一次逆变换的滤波器数量，仅影响显存占用')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader 工作进程数')
    parser.add_argument('--gallery-session', type=int,
                        help='身份检索图库的采集批次，默认与 --session 相反')
    parser.add_argument('--gallery-limit', type=int,
                        help='图库最多取 N 张，默认与 --limit 相同')
    parser.add_argument('--no-identity', action='store_true',
                        help='跳过身份检索与锚点余弦，只做响应/方向码稳定性分析')
    parser.add_argument('--output', type=Path, help='结果输出目录')
    parser.add_argument('--per-image', action='store_true', help='额外保存逐图指标 CSV')
    parser.add_argument('--no-save', action='store_true', help='只打印结果，不写文件')
    return parser.parse_args()


def balanced_subset(samples, limit):
    """从完整清单中均匀抽取 limit 张，保证每个类别都有样本。"""
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
    """读取与评估一致的样本清单，输出 [1, H, W]、取值 [0, 1] 的灰度图。"""
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


def epsilon_of(name):
    """从 eps1 / eps4 这样的目录名解析 epsilon，其他名称返回 None。"""
    match = re.fullmatch(r'eps(\d+(?:\.\d+)?)', name)
    return float(match.group(1)) if match else None


def find_attack_runs(adv_root, model_name, attack, session, epsilon=None):
    """定位一次攻击的全部运行目录，返回 [(运行目录, session 子目录), ...]。

    与低通/位平面脚本不同，这里同时保留 epsN 预算和同目录下的非 epsilon
    运行（例如 cw 目录里的 c1_kappa0，实际是 CW-L2），因此一次 CW 会分析
    ε=1、ε=4 和 L2 三个条件。指定 --epsilon 时只保留对应的 epsN。
    """
    attack_dir = Path(adv_root) / attack / model_name
    if not attack_dir.is_dir():
        return []
    runs = sorted(directory for directory in attack_dir.iterdir()
                  if directory.is_dir() and (directory / f'session{session}').is_dir())
    if not runs:
        return []

    eps_runs = [directory for directory in runs if epsilon_of(directory.name) is not None]
    other_runs = [directory for directory in runs if epsilon_of(directory.name) is None]
    if epsilon is not None:
        eps_runs = [directory for directory in eps_runs
                    if epsilon_of(directory.name) == float(epsilon)]
        selected = sorted(eps_runs, key=lambda directory: epsilon_of(directory.name))
        return [(run, run / f'session{session}') for run in selected]

    if len(other_runs) > 1:
        names = '、'.join(directory.name for directory in other_runs)
        raise ValueError(f'{attack_dir} 有多个非 epsilon 运行（{names}），请手动清理')
    selected = sorted(eps_runs, key=lambda directory: epsilon_of(directory.name))
    selected.extend(other_runs)
    return [(run, run / f'session{session}') for run in selected]


def run_info_of(session_dir):
    """读取一次运行的 run_info.json，失败时返回空字典。"""
    try:
        with (Path(session_dir) / 'run_info.json').open(encoding='utf-8') as file:
            return json.load(file)
    except (OSError, ValueError):
        return {}


def attack_label(attack, run_dir, session_dir):
    """图例名称：L∞ 攻击标注 epsilon，CW-L2 标注 c。"""
    eps = epsilon_of(run_dir.name)
    if eps is not None:
        return f'{attack.upper()}（ε={eps:g}/255）'
    info = run_info_of(session_dir)
    params = info.get('params', {})
    attack_name = info.get('attack', attack)
    if attack_name == 'cw-l2' or ('c' in params and 'epsilon' not in params):
        return f"CW-L2（c={params.get('c', 1.0):g}）"
    return attack.upper()


def run_short_name(attack, run_dir, session_dir):
    """表格/图例里使用的短名称。"""
    eps = epsilon_of(run_dir.name)
    if eps is not None:
        return f'{attack.upper()} ε{eps:g}'
    info = run_info_of(session_dir)
    if info.get('attack') == 'cw-l2' or (
            'c' in info.get('params', {}) and 'epsilon' not in info.get('params', {})):
        return 'CW-L2'
    return attack.upper()


def structure_of(wavelength):
    """按波长给出粗略的掌纹结构标签，仅用于表格分组。"""
    if wavelength < 8:
        return '细粒度纹理'
    if wavelength < 16:
        return '褶皱'
    return '掌纹主线'


def new_accumulator(shape):
    return {
        'sum': np.zeros(shape, dtype=np.float64),
        'sumsq': np.zeros(shape, dtype=np.float64),
        'count': np.zeros(shape, dtype=np.float64),
    }


def accumulate(accumulator, values):
    """把一批逐图指标累加到 sum/sumsq/count，自动忽略 NaN。"""
    array = values.detach().double().cpu().numpy()
    valid = np.isfinite(array)
    array = np.where(valid, array, 0.0)
    accumulator['sum'] += array.sum(axis=0)
    accumulator['sumsq'] += (array * array).sum(axis=0)
    accumulator['count'] += valid.sum(axis=0)


def finish_accumulator(accumulator):
    """返回 (mean, std)；没有有效样本的位置为 NaN。"""
    count = accumulator['count']
    mean = np.divide(accumulator['sum'], count,
                     out=np.full_like(count, np.nan), where=count > 0)
    second = np.divide(accumulator['sumsq'], count,
                       out=np.full_like(count, np.nan), where=count > 0)
    variance = np.clip(second - mean ** 2, 0.0, None)
    return mean, np.sqrt(variance)


def descriptor(histogram, feature):
    """从 [N, S, G, G, O] 分块直方图中取出一个特征集并做 L2 归一化。"""
    count = histogram.shape[0]
    kind = feature['kind']
    if kind == 'joint':
        values = histogram.reshape(count, -1)
    elif kind == 'scale':
        values = histogram[:, feature['scale']].reshape(count, -1)
    elif kind == 'orientation':
        values = histogram[:, :, :, :, feature['orientation']].reshape(count, -1)
    elif kind == 'cell':
        values = histogram[:, feature['scale'], :, :, feature['orientation']].reshape(
            count, -1)
    else:
        raise ValueError(f'未知特征类型 {kind!r}')
    return F.normalize(values, dim=1)


def feature_sets(bank):
    """枚举身份分析使用的特征集合：全局、单尺度、单方向、单尺度单方向。"""
    scales = bank.wavelengths.reshape(bank.num_scales, bank.num_orientations)[:, 0]
    angles = bank.scale_angles
    features = [{
        'key': 'joint', 'kind': 'joint', 'scale': None, 'orientation': None,
        'label': '全部尺度×方向',
    }]
    for index in range(bank.num_scales):
        features.append({
            'key': f'scale-{index}', 'kind': 'scale', 'scale': index,
            'orientation': None,
            'label': f'尺度 λ={float(scales[index]):g}',
        })
    for index in range(bank.num_orientations):
        features.append({
            'key': f'orientation-{index}', 'kind': 'orientation', 'scale': None,
            'orientation': index,
            'label': f'方向 θ={float(angles[index]):g}°',
        })
    for scale in range(bank.num_scales):
        for orientation in range(bank.num_orientations):
            features.append({
                'key': f'cell-{scale}-{orientation}', 'kind': 'cell',
                'scale': scale, 'orientation': orientation,
                'label': (f'λ={float(scales[scale]):g}, '
                          f'θ={float(angles[orientation]):g}°'),
            })
    return features


@torch.no_grad()
def collect_histograms(dataset, bank, device, batch_size, num_workers, grid,
                       filter_chunk, dominance, desc=''):
    """对每个条件提取方向码分块直方图，返回 (hist [N,S,G,G,O], labels [N])。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    histograms, labels = [], []
    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        magnitude = gabor_magnitude(images, bank, filter_chunk)
        codes = dominant_orientation(magnitude, bank, dominance)
        histograms.append(orientation_histogram(codes, bank, grid).cpu())
        labels.append(targets)
    return torch.cat(histograms), torch.cat(labels)


@torch.no_grad()
def rank_metrics(query, query_labels, gallery, gallery_labels, topk=5, chunk=2048):
    """余弦检索的 Rank-1/Rank-5，query 与 gallery 都已 L2 归一化。"""
    correct1 = correct5 = 0
    for start in range(0, query.shape[0], chunk):
        similarity = query[start:start + chunk] @ gallery.T
        index = similarity.topk(topk, dim=1).indices
        predicted = gallery_labels[index]
        target = query_labels[start:start + chunk, None]
        correct1 += (predicted[:, 0] == target[:, 0]).sum().item()
        correct5 += (predicted == target).any(dim=1).sum().item()
    total = query.shape[0]
    return correct1 / total, correct5 / total


@torch.no_grad()
def identity_rows(clean_hist, adv_hist, query_labels, gallery_descriptors,
                  gallery_labels, features, n_gallery, bank, clean_rank1=None):
    """计算一个条件的身份检索与锚点稳定性，返回 (DataFrame, rank1 字典)。"""
    rows = []
    rank1_values = {}
    for feature in features:
        clean_desc = descriptor(clean_hist, feature)
        adv_desc = descriptor(adv_hist, feature)
        anchor = (clean_desc * adv_desc).sum(dim=1)
        rank1, rank5 = rank_metrics(
            adv_desc, query_labels, gallery_descriptors[feature['key']],
            gallery_labels)
        baseline = rank1 if clean_rank1 is None else clean_rank1[feature['key']]
        rank1_values[feature['key']] = rank1
        rows.append({
            'feature_key': feature['key'],
            'feature_kind': feature['kind'],
            'feature_label': feature['label'],
            'scale': feature['scale'],
            'orientation_index': feature['orientation'],
            'descriptor_dim': adv_desc.shape[1],
            'images': adv_desc.shape[0],
            'gallery_images': n_gallery,
            'anchor_cosine_mean': anchor.mean().item(),
            'anchor_cosine_std': anchor.std(unbiased=False).item(),
            'rank1': rank1,
            'rank5': rank5,
            'rank1_clean': baseline,
            'rank1_delta': rank1 - baseline,
        })
    frame = pd.DataFrame(rows)
    if 'scale' in frame:
        scales = torch.as_tensor(
            [float(bank.wavelengths.reshape(bank.num_scales,
                                            bank.num_orientations)[s, 0])
             for s in range(bank.num_scales)])
        frame['wavelength'] = frame['scale'].map(
            lambda value: float(scales[int(value)]) if pd.notna(value) else np.nan)
    if 'orientation_index' in frame:
        angles = bank.scale_angles.cpu().numpy()
        frame['orientation_deg'] = frame['orientation_index'].map(
            lambda value: float(angles[int(value)]) if pd.notna(value) else np.nan)
    # 列顺序：先特征信息，再指标。
    columns = ['feature_key', 'feature_kind', 'feature_label', 'scale',
               'wavelength', 'orientation_index', 'orientation_deg',
               'descriptor_dim', 'images', 'gallery_images',
               'anchor_cosine_mean', 'anchor_cosine_std', 'rank1', 'rank5',
               'rank1_clean', 'rank1_delta']
    return frame[columns], rank1_values


def clean_frames(bank, image_count):
    """干净样本的基准表：响应完全一致、方向码完全保留。"""
    scales = bank.wavelengths.reshape(bank.num_scales, bank.num_orientations)[:, 0]
    angles = bank.scale_angles.cpu().numpy()
    scale_rows, long_rows = [], []
    for scale in range(bank.num_scales):
        wavelength = float(scales[scale])
        scale_rows.append({
            'scale': scale,
            'wavelength': wavelength,
            'structure': structure_of(wavelength),
            'images': image_count,
            'code_agreement_mean': 1.0,
            'code_agreement_std': 0.0,
            'angular_error_deg_mean': 0.0,
            'angular_error_deg_std': 0.0,
            'magnitude_corr_mean': 1.0,
            'magnitude_corr_std': 0.0,
            'energy_ratio_db_mean': 0.0,
            'energy_ratio_db_std': 0.0,
            'code_retention_pooled': 1.0,
            'support': np.nan,
        })
        for orientation in range(bank.num_orientations):
            long_rows.append({
                'scale': scale,
                'wavelength': wavelength,
                'structure': structure_of(wavelength),
                'orientation_index': orientation,
                'orientation_deg': float(angles[orientation]),
                'images': image_count,
                'magnitude_corr_mean': 1.0,
                'magnitude_corr_std': 0.0,
                'energy_ratio_db_mean': 0.0,
                'energy_ratio_db_std': 0.0,
                'code_retention_mean': 1.0,
                'code_retention_std': 0.0,
                'code_retention_pooled': 1.0,
                'support': np.nan,
            })
    return pd.DataFrame(scale_rows), pd.DataFrame(long_rows)


@torch.no_grad()
def paired_condition(clean_dataset, adv_dataset, clean_hist, bank, settings,
                     device, desc=''):
    """对比干净与对抗条件，返回汇总表、对抗方向码直方图和转移计数。"""
    clean_loader = DataLoader(clean_dataset, batch_size=settings['batch_size'],
                              shuffle=False, num_workers=settings['num_workers'])
    adv_loader = DataLoader(adv_dataset, batch_size=settings['batch_size'],
                            shuffle=False, num_workers=settings['num_workers'])
    if len(clean_loader) != len(adv_loader):
        raise ValueError('干净与对抗样本数量不一致')

    scales, orientations = bank.num_scales, bank.num_orientations
    corr_acc = new_accumulator((scales, orientations))
    energy_acc = new_accumulator((scales, orientations))
    retention_acc = new_accumulator((scales, orientations))
    agreement_acc = new_accumulator((scales,))
    angle_acc = new_accumulator((scales,))
    transition_sum = np.zeros((scales, orientations, orientations), dtype=np.float64)
    count_sum = np.zeros((scales, orientations), dtype=np.float64)
    adv_histograms = []
    per_image_rows = []
    seen = 0

    joint_feature = {
        'key': 'joint', 'kind': 'joint', 'scale': None, 'orientation': None,
        'label': '全部尺度×方向',
    }

    for (clean_images, clean_labels), (adv_images, adv_labels) in tqdm(
            zip(clean_loader, adv_loader), total=len(clean_loader), desc=desc,
            leave=False):
        if not torch.equal(clean_labels, adv_labels):
            raise ValueError('干净与对抗样本的标签顺序不一致')
        clean_images = clean_images.to(device)
        adv_images = adv_images.to(device)

        clean_mag = gabor_magnitude(clean_images, bank, settings['filter_chunk'])
        adv_mag = gabor_magnitude(adv_images, bank, settings['filter_chunk'])
        clean_codes = dominant_orientation(clean_mag, bank, settings['dominance'])
        adv_codes = dominant_orientation(adv_mag, bank, settings['dominance'])

        correlation = magnitude_correlation(clean_mag, adv_mag).reshape(
            -1, scales, orientations)
        energy = energy_ratio_db(clean_mag, adv_mag).reshape(
            -1, scales, orientations)
        agreement = code_agreement(clean_codes, adv_codes)
        angular = angular_error_deg(clean_codes, adv_codes, bank)
        transition, counts = code_transition(clean_codes, adv_codes, bank)
        retention = transition.diagonal(dim1=-2, dim2=-1) / counts.clamp_min(1.0)
        retention = torch.where(counts > 0, retention,
                                torch.full_like(retention, float('nan')))

        accumulate(corr_acc, correlation)
        accumulate(energy_acc, energy)
        accumulate(retention_acc, retention)
        accumulate(agreement_acc, agreement)
        accumulate(angle_acc, angular)
        transition_sum += transition.sum(dim=0).double().cpu().numpy()
        count_sum += counts.sum(dim=0).double().cpu().numpy()

        adv_histograms.append(orientation_histogram(
            adv_codes, bank, settings['code_grid']).cpu())

        if settings['per_image']:
            batch = clean_images.shape[0]
            frame = pd.DataFrame({
                'filename': [Path(path).name
                             for path, _ in clean_dataset.samples[seen:seen + batch]],
                'label': clean_labels.numpy(),
            })
            agreement_np = agreement.double().cpu().numpy()
            angular_np = angular.double().cpu().numpy()
            correlation_np = correlation.double().cpu().numpy()
            energy_np = energy.double().cpu().numpy()
            for scale in range(scales):
                wavelength = float(bank.wavelengths.reshape(
                    scales, orientations)[scale, 0])
                frame[f'agree_l{wavelength:g}'] = agreement_np[:, scale]
                frame[f'angle_l{wavelength:g}'] = angular_np[:, scale]
                frame[f'corr_l{wavelength:g}'] = correlation_np[:, scale, :].mean(axis=1)
                frame[f'energy_db_l{wavelength:g}'] = energy_np[:, scale, :].mean(axis=1)
            if clean_hist is not None:
                clean_slice = clean_hist[seen:seen + batch].to(device)
                anchor = (descriptor(clean_slice, joint_feature)
                          * descriptor(adv_histograms[-1].to(device), joint_feature)
                          ).sum(dim=1).cpu().numpy()
                frame['anchor_joint'] = anchor
            per_image_rows.append(frame)
        seen += clean_images.shape[0]

    correlation_mean, correlation_std = finish_accumulator(corr_acc)
    energy_mean, energy_std = finish_accumulator(energy_acc)
    retention_mean, retention_std = finish_accumulator(retention_acc)
    agreement_mean, agreement_std = finish_accumulator(agreement_acc)
    angular_mean, angular_std = finish_accumulator(angle_acc)

    diag = np.diagonal(transition_sum, axis1=1, axis2=2)      # [S, O]
    pooled = np.divide(diag, count_sum,
                       out=np.full_like(diag, np.nan), where=count_sum > 0)

    long_rows = []
    for scale in range(scales):
        wavelength = float(bank.wavelengths.reshape(
            scales, orientations)[scale, 0])
        for orientation in range(orientations):
            long_rows.append({
                'scale': scale,
                'wavelength': wavelength,
                'structure': structure_of(wavelength),
                'orientation_index': orientation,
                'orientation_deg': float(bank.scale_angles[orientation]),
                'images': seen,
                'magnitude_corr_mean': correlation_mean[scale, orientation],
                'magnitude_corr_std': correlation_std[scale, orientation],
                'energy_ratio_db_mean': energy_mean[scale, orientation],
                'energy_ratio_db_std': energy_std[scale, orientation],
                'code_retention_mean': retention_mean[scale, orientation],
                'code_retention_std': retention_std[scale, orientation],
                'code_retention_pooled': pooled[scale, orientation],
                'support': count_sum[scale, orientation],
            })

    scale_rows = []
    for scale in range(scales):
        wavelength = float(bank.wavelengths.reshape(
            scales, orientations)[scale, 0])
        scale_pooled = np.divide(diag[scale].sum(), count_sum[scale].sum()) \
            if count_sum[scale].sum() > 0 else np.nan
        scale_rows.append({
            'scale': scale,
            'wavelength': wavelength,
            'structure': structure_of(wavelength),
            'images': seen,
            'code_agreement_mean': agreement_mean[scale],
            'code_agreement_std': agreement_std[scale],
            'angular_error_deg_mean': angular_mean[scale],
            'angular_error_deg_std': angular_std[scale],
            'magnitude_corr_mean': correlation_mean[scale].mean(),
            'magnitude_corr_std': correlation_std[scale].mean(),
            'energy_ratio_db_mean': energy_mean[scale].mean(),
            'energy_ratio_db_std': energy_std[scale].mean(),
            'code_retention_pooled': scale_pooled,
            'support': count_sum[scale].sum(),
        })

    return {
        'scale': pd.DataFrame(scale_rows),
        'long': pd.DataFrame(long_rows),
        'adv_hist': torch.cat(adv_histograms),
        'transition': transition_sum,
        'counts': count_sum,
        'images': seen,
        'per_image': pd.concat(per_image_rows, ignore_index=True)
        if per_image_rows else None,
    }


def tag_frame(frame, model, architecture, condition, run, dominance, session):
    """给结果表插入模型/条件等上下文列。"""
    frame = frame.copy()
    metadata = {
        'model': model,
        'architecture': architecture,
        'condition': condition,
        'run': run,
        'dominance': dominance,
        'session': session,
    }
    for key, value in reversed(list(metadata.items())):
        frame.insert(0, key, value)
    return frame


def print_condition_table(model, series):
    """打印每个条件的总体稳定性。"""
    print(f'\n[{model}] 条件汇总（码一致/码保留为尺度平均，相关为全部尺度×方向平均）')
    header = (f"{'条件':<18}{'样本':>7}{'码一致':>10}{'码保留':>10}"
              f"{'响应相关':>12}{'锚点余弦':>12}{'Rank-1':>10}")
    print(header)
    print('-' * len(header))
    for item in series:
        long = item['long']
        scale = item['scale']
        identity = item['identity']
        joint = None if identity is None else identity[identity.feature_kind == 'joint']
        agreement = scale['code_agreement_mean'].mean()
        retention = long['code_retention_pooled'].mean()
        correlation = long['magnitude_corr_mean'].mean()
        anchor = joint['anchor_cosine_mean'].iloc[0] if joint is not None else np.nan
        rank1 = joint['rank1'].iloc[0] if joint is not None else np.nan
        anchor_text = f'{anchor:>12.3f}' if np.isfinite(anchor) else f"{'-':>12}"
        rank_text = f'{rank1 * 100:>9.2f}%' if np.isfinite(rank1) else f"{'-':>10}"
        print(f"{item['short']:<18}{item['images']:>7}{agreement:>10.3f}"
              f"{retention:>10.3f}{correlation:>12.3f}{anchor_text}{rank_text}")


def print_scale_table(model, series, bank):
    """打印方向码一致率随尺度的变化。"""
    wavelengths = bank.wavelengths.reshape(
        bank.num_scales, bank.num_orientations)[:, 0]
    header = f"{'条件':<18}" + ''.join(f'{float(w):>8g}' for w in wavelengths)
    print(f'\n[{model}] 方向码一致率（行=条件，列=λ）')
    print(header)
    print('-' * len(header))
    for item in series:
        values = item['scale'].sort_values('scale')['code_agreement_mean'].to_numpy()
        print(f"{item['short']:<18}" + ''.join(f'{value:>8.3f}' for value in values))


def print_orientation_table(model, series, bank):
    """打印方向码保留率随方向的变化（各尺度等权平均）。"""
    angles = bank.scale_angles.cpu().numpy()
    header = f"{'条件':<18}" + ''.join(f'{float(a):>8g}' for a in angles)
    print(f'\n[{model}] 方向码保留率（行=条件，列=θ，尺度等权平均）')
    print(header)
    print('-' * len(header))
    for item in series:
        long = item['long']
        values = long.groupby('orientation_index')['code_retention_pooled'].mean()
        print(f"{item['short']:<18}"
              + ''.join(f'{values.get(index, np.nan):>8.3f}'
                        for index in range(bank.num_orientations)))


def print_identity_table(model, series, bank):
    """打印按尺度拆开的 Rank-1。"""
    wavelengths = bank.wavelengths.reshape(
        bank.num_scales, bank.num_orientations)[:, 0]
    header = f"{'条件':<18}" + ''.join(f'{float(w):>8g}' for w in wavelengths) \
        + f"{'联合':>8}"
    print(f'\n[{model}] 身份检索 Rank-1（%，图库为另一 session；列=λ）')
    print(header)
    print('-' * len(header))
    for item in series:
        identity = item['identity']
        if identity is None:
            continue
        cells = identity[identity.feature_kind == 'scale'].sort_values('scale')
        joint = identity[identity.feature_kind == 'joint']['rank1'].iloc[0]
        line = f"{item['short']:<18}"
        line += ''.join(f'{value * 100:>8.2f}' for value in cells['rank1'])
        line += f'{joint * 100:>8.2f}'
        print(line)


def setup_matplotlib():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def save_figure(figure, prefix):
    png_path = prefix.with_suffix('.png')
    pdf_path = prefix.with_suffix('.pdf')
    figure.savefig(png_path, dpi=220, bbox_inches='tight')
    figure.savefig(pdf_path, bbox_inches='tight')
    plt.close(figure)
    return png_path


def plot_heatmaps(model, series, bank, column, title, colorbar_label,
                  prefix, vmin, vmax):
    """每个对抗条件一张尺度×方向热图。"""
    attacks = [item for item in series if item['condition'] != 'clean']
    if not attacks:
        return None
    if vmin is None:
        values = np.concatenate(
            [item['long'][column].to_numpy() for item in attacks])
        values = values[np.isfinite(values)]
        vmin = min(0.0, float(np.floor(values.min() * 10) / 10)) if values.size else 0.0
    ncols = min(3, len(attacks))
    nrows = int(np.ceil(len(attacks) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows),
                                squeeze=False)
    figure.subplots_adjust(hspace=0.42, wspace=0.22)
    wavelengths = bank.wavelengths.reshape(
        bank.num_scales, bank.num_orientations)[:, 0]
    angles = bank.scale_angles.cpu().numpy()
    image = None
    for index, (axis, item) in enumerate(zip(axes.flat, attacks)):
        row, col = divmod(index, ncols)
        data = item['long'][column].to_numpy().reshape(
            bank.num_scales, bank.num_orientations)
        image = axis.imshow(data, vmin=vmin, vmax=vmax, cmap='viridis',
                            aspect='auto')
        axis.set_xticks(range(bank.num_orientations))
        axis.set_xticklabels([f'{float(angle):g}°' for angle in angles],
                             fontsize=8)
        axis.set_yticks(range(bank.num_scales))
        axis.set_yticklabels([f'{float(value):g}' for value in wavelengths],
                             fontsize=8)
        if row == nrows - 1:
            axis.set_xlabel('方向 θ')
        if col == 0:
            axis.set_ylabel('波长 λ')
        axis.set_title(item['short'], fontsize=10)
        midpoint = (vmin + vmax) / 2
        for row in range(bank.num_scales):
            for col in range(bank.num_orientations):
                value = data[row, col]
                axis.text(col, row, f'{value:.2f}', ha='center', va='center',
                          fontsize=7, color='white' if value < midpoint else 'black')
    for axis in axes.flat[len(attacks):]:
        axis.axis('off')
    figure.suptitle(f'{model}：{title}', fontsize=13)
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9,
                    label=colorbar_label)
    return save_figure(figure, prefix)


def plot_curves(model, series, bank, prefix):
    """方向码一致率、方向保留率、响应相关性随尺度/方向的变化。"""
    wavelengths = bank.wavelengths.reshape(
        bank.num_scales, bank.num_orientations)[:, 0].cpu().numpy()
    angles = bank.scale_angles.cpu().numpy()
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.4))
    for item in series:
        common = {
            'color': item['color'], 'linestyle': item['style'],
            'marker': item['marker'], 'markersize': 4.5,
            'linewidth': 2.6 if item['condition'] == 'clean' else 2.0,
            'label': item['label'], 'dash_capstyle': 'butt',
        }
        scale = item['scale'].sort_values('scale')
        axes[0].plot(wavelengths, scale['code_agreement_mean'].to_numpy(), **common)
        long = item['long']
        correlation = long['magnitude_corr_mean'].to_numpy().reshape(
            bank.num_scales, bank.num_orientations)
        retention = long['code_retention_pooled'].to_numpy().reshape(
            bank.num_scales, bank.num_orientations)
        axes[1].plot(angles, retention.mean(axis=0), **common)
        axes[2].plot(wavelengths, correlation.mean(axis=1), **common)

    axes[0].set_xlabel('Gabor 波长 λ（像素）')
    axes[0].set_ylabel('方向码一致率')
    axes[0].set_title('方向码一致率随尺度变化')
    axes[1].set_xlabel('方向 θ（度）')
    axes[1].set_ylabel('方向码保留率（尺度平均）')
    axes[1].set_title('方向码保留率随方向变化')
    axes[2].set_xlabel('Gabor 波长 λ（像素）')
    axes[2].set_ylabel('干净/对抗响应相关性')
    axes[2].set_title('方向响应相关性随尺度变化')
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center',
                  ncol=min(5, len(labels)), fontsize=8, frameon=False)
    figure.suptitle(f'{model}：多尺度多方向 Gabor 方向码稳定性', fontsize=13)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    return save_figure(figure, prefix)


def identity_matrix(identity, bank):
    """把单尺度单方向的行透视成 [S, O] 的 Rank-1 矩阵。"""
    cells = identity[identity.feature_kind == 'cell']
    matrix = cells.pivot(index='scale', columns='orientation_index', values='rank1')
    return matrix.reindex(index=range(bank.num_scales),
                          columns=range(bank.num_orientations)).to_numpy()


def plot_identity(model, series, bank, prefix):
    """Rank-1/锚点余弦随尺度变化，以及尺度×方向的 Rank-1 热图。"""
    with_identity = [item for item in series if item['identity'] is not None]
    if not with_identity:
        return None
    clean = series[0]
    attacks = [item for item in series[1:] if item['identity'] is not None]
    worst = None
    if attacks:
        worst = min(
            attacks,
            key=lambda item: item['identity'][
                item['identity'].feature_kind == 'joint']['rank1'].iloc[0])

    wavelengths = bank.wavelengths.reshape(
        bank.num_scales, bank.num_orientations)[:, 0].cpu().numpy()
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))
    for item in with_identity:
        identity = item['identity']
        scale_rows = identity[identity.feature_kind == 'scale'].sort_values('scale')
        common = {
            'color': item['color'], 'linestyle': item['style'],
            'marker': item['marker'], 'markersize': 4.5,
            'linewidth': 2.6 if item['condition'] == 'clean' else 2.0,
            'label': item['label'], 'dash_capstyle': 'butt',
        }
        axes[0, 0].plot(wavelengths, scale_rows['rank1'].to_numpy(), **common)
        axes[0, 1].plot(wavelengths, scale_rows['anchor_cosine_mean'].to_numpy(),
                        **common)

    axes[0, 0].set_xlabel('Gabor 波长 λ（像素）')
    axes[0, 0].set_ylabel('Rank-1 识别率')
    axes[0, 0].set_title('身份检索 Rank-1 随尺度变化')
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 1].set_xlabel('Gabor 波长 λ（像素）')
    axes[0, 1].set_ylabel('干净/对抗描述子余弦')
    axes[0, 1].set_title('身份锚点余弦随尺度变化')
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].grid(alpha=0.3)

    angles = bank.scale_angles.cpu().numpy()
    clean_matrix = identity_matrix(clean['identity'], bank)
    worst_item = worst if worst is not None else clean
    worst_matrix = identity_matrix(worst_item['identity'], bank)
    drop = clean_matrix - worst_matrix
    panels = [
        (axes[1, 0], clean_matrix, 'viridis',
         f'干净样本：单尺度单方向 Rank-1', 'Rank-1 识别率', False),
        (axes[1, 1], drop, 'Reds',
         (f'Rank-1 下降（干净 - {worst_item["short"]}）'
          if worst is not None else '无对抗条件'),
         'Rank-1 下降（比例）', True),
    ]
    images = []
    for axis, matrix, cmap, caption, colorbar_label, is_drop in panels:
        limit = np.nanmax(matrix) if np.isfinite(matrix).any() else 1.0
        limit = max(0.01, float(limit))
        image = axis.imshow(matrix, vmin=0.0, vmax=limit, cmap=cmap,
                            aspect='auto')
        images.append((axis, image, colorbar_label))
        axis.set_xticks(range(bank.num_orientations))
        axis.set_xticklabels([f'{float(angle):g}°' for angle in angles], fontsize=8)
        axis.set_yticks(range(bank.num_scales))
        axis.set_yticklabels([f'{float(value):g}' for value in wavelengths], fontsize=8)
        axis.set_xlabel('方向 θ')
        axis.set_ylabel('波长 λ')
        axis.set_title(caption, fontsize=10)
        midpoint = limit / 2
        for row in range(bank.num_scales):
            for col in range(bank.num_orientations):
                value = matrix[row, col]
                # viridis 低值偏暗、Reds 低值偏亮，文字颜色相应反转。
                if is_drop:
                    color = 'black' if value < midpoint else 'white'
                else:
                    color = 'white' if value < midpoint else 'black'
                axis.text(col, row, f'{value * 100:.0f}', ha='center', va='center',
                          fontsize=7, color=color)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center',
                  bbox_to_anchor=(0.5, 0.005), ncol=min(5, len(labels)),
                  fontsize=8, frameon=False)
    figure.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.12,
                           wspace=0.32, hspace=0.38)
    for axis, image, colorbar_label in images:
        position = axis.get_position()
        colorbar_axis = figure.add_axes(
            [position.x1 + 0.008, position.y0, 0.010, position.height])
        figure.colorbar(image, cax=colorbar_axis, label=colorbar_label)
    title = f'{model}：Gabor 方向码的身份检索稳定性'
    if worst is not None:
        title += f'（最差条件：{worst["short"]}）'
    figure.suptitle(title, fontsize=13, y=0.97)
    return save_figure(figure, prefix)


def plot_transitions(model, series, bank, prefix):
    """每个对抗条件一张方向码转移矩阵（各尺度汇总）。"""
    attacks = [item for item in series
               if item['condition'] != 'clean' and item['transition'] is not None]
    if not attacks:
        return None
    ncols = min(3, len(attacks))
    nrows = int(np.ceil(len(attacks) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.3 * nrows),
                                squeeze=False)
    figure.subplots_adjust(hspace=0.42, wspace=0.22)
    angles = bank.scale_angles.cpu().numpy()
    image = None
    for index, (axis, item) in enumerate(zip(axes.flat, attacks)):
        row, col = divmod(index, ncols)
        transition = item['transition'].sum(axis=0)            # [O, O]
        counts = item['counts'].sum(axis=0)                    # [O]
        probability = np.divide(
            transition, counts[:, None],
            out=np.zeros_like(transition), where=counts[:, None] > 0)
        image = axis.imshow(probability, vmin=0.0, vmax=1.0, cmap='Blues',
                            aspect='auto')
        axis.set_xticks(range(bank.num_orientations))
        axis.set_xticklabels([f'{float(angle):g}°' for angle in angles], fontsize=8)
        axis.set_yticks(range(bank.num_orientations))
        axis.set_yticklabels([f'{float(angle):g}°' for angle in angles], fontsize=8)
        if row == nrows - 1:
            axis.set_xlabel('对抗方向码')
        if col == 0:
            axis.set_ylabel('干净方向码')
        axis.set_title(item['short'], fontsize=10)
        for row in range(bank.num_orientations):
            for col in range(bank.num_orientations):
                value = probability[row, col]
                axis.text(col, row, f'{value:.2f}', ha='center', va='center',
                          fontsize=7, color='white' if value > 0.5 else 'black')
    for axis in axes.flat[len(attacks):]:
        axis.axis('off')
    figure.suptitle(f'{model}：方向码转移概率 P(对抗码 | 干净码)（尺度汇总）',
                    fontsize=13)
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.9, label='条件概率')
    return save_figure(figure, prefix)


def transition_frame(series, bank, model, architecture, dominance, session):
    """把转移矩阵整理成一行一个 from→to 的长表。"""
    rows = []
    angles = bank.scale_angles.cpu().numpy()
    wavelengths = bank.wavelengths.reshape(
        bank.num_scales, bank.num_orientations)[:, 0]
    condition = series[0]['condition'] if series else ''
    run = series[0]['run'] if series else ''
    for item in series:
        if item['transition'] is None:
            continue
        blocks = [(scale, item['transition'][scale], item['counts'][scale])
                  for scale in range(bank.num_scales)]
        blocks.append(('overall', item['transition'].sum(axis=0),
                       item['counts'].sum(axis=0)))
        for scale, transition, counts in blocks:
            for source in range(bank.num_orientations):
                support = counts[source]
                for target in range(bank.num_orientations):
                    probability = (transition[source, target] / support
                                   if support > 0 else np.nan)
                    rows.append({
                        'scale': scale,
                        'wavelength': (float(wavelengths[scale])
                                       if scale != 'overall' else np.nan),
                        'from_index': source,
                        'from_deg': float(angles[source]),
                        'to_index': target,
                        'to_deg': float(angles[target]),
                        'probability': probability,
                        'support': support,
                    })
    frame = pd.DataFrame(rows)
    return tag_frame(frame, model, architecture, condition, run,
                     dominance, session)


def main():
    args = parse_args()
    setup_matplotlib()
    if args.wavelengths is None or len(args.wavelengths) == 0:
        raise SystemExit('--wavelengths 至少需要一个波长')
    if args.code_grid < 1:
        raise SystemExit('--code-grid 必须为正整数')
    if args.batch_size < 1:
        raise SystemExit('--batch-size 必须为正整数')
    if args.filter_chunk < 1:
        raise SystemExit('--filter-chunk 必须为正整数')

    config = load_config(args.config)
    settings = config['evaluation']
    attack_settings = config.get('attacks', {})

    session = args.session if args.session is not None else settings.get('session', 2)
    data_root = args.data_root or resolve_path(settings.get('data_root', 'data/resources'))
    adv_root = args.adv_root or resolve_path(attack_settings.get('output_dir', 'outputs/adv'))
    output_dir = args.output or resolve_path('results/gabor_robustness')
    names = args.model or list(config['models'])
    attacks = args.attack or list(DEFAULT_ATTACKS)
    device = select_device(args.device)

    unknown = [name for name in names if name not in config['models']]
    if unknown:
        raise SystemExit(f'配置中没有模型 {", ".join(unknown)}；'
                         f'可选 {", ".join(config["models"])}')

    orientations = args.angles if args.angles is not None else args.orientations
    bank = make_gabor_bank(
        IMAGE_SIZE, IMAGE_SIZE, wavelengths=args.wavelengths,
        orientations=orientations, sigma_ratio=args.sigma_ratio,
        gamma=args.gamma, pad_ratio=args.pad_ratio, device=device)
    features = feature_sets(bank)
    scale_wavelengths = bank.wavelengths.reshape(
        bank.num_scales, bank.num_orientations)[:, 0].cpu().numpy()
    angle_values = bank.scale_angles.cpu().numpy()

    gallery_session = args.gallery_session
    if gallery_session is None:
        gallery_session = 2 if session == 1 else 1
    if args.no_identity:
        gallery_session = None
    elif gallery_session == session:
        print(f'警告：图库与查询都使用 session{session}，可能包含同一张图。')

    print(f'设备：{device}，数据：{data_root}（session{session}）')
    print(f'Gabor：{bank.num_scales} 尺度 × {bank.num_orientations} 方向 = '
          f'{bank.num_filters} 个滤波器；λ={", ".join(f"{v:g}" for v in scale_wavelengths)}；'
          f'θ={", ".join(f"{v:g}°" for v in angle_values)}')
    print(f'主方向规则：{args.dominance}；编码网格：{args.code_grid}×{args.code_grid}；'
          f'填充：{bank.pad} px（{bank.padded_height}×{bank.padded_width}）')

    clean_dataset = make_pixel_dataset(
        session, data_root, limit=args.limit, balance_classes=args.balance_classes)
    covered = len({label for _, label in clean_dataset.samples})
    print(f'干净样本：{len(clean_dataset)} 张，覆盖 {covered} 类')

    gallery_dataset = None
    gallery_labels = None
    gallery_descriptors = None
    clean_hist = None
    clean_identity = None
    clean_rank1 = None
    if not args.no_identity:
        gallery_limit = args.gallery_limit if args.gallery_limit is not None else args.limit
        gallery_dataset = make_pixel_dataset(
            gallery_session, data_root, limit=gallery_limit,
            balance_classes=args.balance_classes)
        gallery_covered = len({label for _, label in gallery_dataset.samples})
        print(f'身份图库：session{gallery_session}，{len(gallery_dataset)} 张，'
              f'覆盖 {gallery_covered} 类')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_info = {
            'session': session,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'dominance': args.dominance,
            'wavelengths': [float(value) for value in scale_wavelengths],
            'orientations_deg': [float(value) for value in angle_values],
            'sigma_ratio': args.sigma_ratio,
            'gamma': args.gamma,
            'pad_ratio': args.pad_ratio,
            'pad': bank.pad,
            'code_grid': args.code_grid,
            'limit': args.limit,
            'balance_classes': args.balance_classes,
            'gallery_session': gallery_session,
            'gallery_limit': args.gallery_limit,
            'attacks': attacks,
            'epsilon': args.epsilon,
            'models': names,
            'images': len(clean_dataset),
            'device': str(device),
        }
        with (output_dir / f'session{session}_{args.dominance}_{stamp}_run_info.json').open(
                'w', encoding='utf-8') as file:
            json.dump(run_info, file, ensure_ascii=False, indent=2)

    clean_scale, clean_long = clean_frames(bank, len(clean_dataset))

    # 干净样本的方向码直方图与身份基线只需算一次。
    if not args.no_identity:
        print('\n提取干净查询集与图库的 Gabor 方向码直方图 ...')
        clean_hist, clean_labels = collect_histograms(
            clean_dataset, bank, device, args.batch_size, args.num_workers,
            args.code_grid, args.filter_chunk, args.dominance,
            desc='clean-query')
        gallery_hist, gallery_labels = collect_histograms(
            gallery_dataset, bank, device, args.batch_size, args.num_workers,
            args.code_grid, args.filter_chunk, args.dominance,
            desc='gallery')
        clean_hist_device = clean_hist.to(device)
        gallery_hist_device = gallery_hist.to(device)
        gallery_descriptors = {
            feature['key']: descriptor(gallery_hist_device, feature)
            for feature in features
        }
        clean_labels_device = clean_labels.to(device)
        gallery_labels_device = gallery_labels.to(device)
        clean_identity, clean_rank1 = identity_rows(
            clean_hist_device, clean_hist_device, clean_labels_device,
            gallery_descriptors, gallery_labels_device, features,
            len(gallery_dataset), bank)
        print(f'干净基线 Rank-1（联合特征）：'
              f'{clean_identity[clean_identity.feature_kind == "joint"]["rank1"].iloc[0] * 100:.2f}%')

    robustness_frames = []
    scale_frames = []
    identity_frames = []
    transition_frames = []
    per_image_frames = []

    for name in names:
        entry = config['models'][name]
        architecture = entry.get('arch')
        print(f"\n[{name}] {architecture}")
        conditions = [('clean', '干净', '干净样本', clean_dataset,
                       CLEAN_COLOR, '-', 's', 'clean')]

        for attack in attacks:
            found = find_attack_runs(adv_root, name, attack, session, args.epsilon)
            if not found:
                detail = f'eps{args.epsilon:g}' if args.epsilon is not None else '可用'
                print(f'  跳过 {attack}：{adv_root / attack / name} 没有 {detail} 的 '
                      f'session{session} 运行')
                continue
            eps_values = sorted(epsilon_of(run.name) for run, _ in found
                                if epsilon_of(run.name) is not None)
            budget_styles = ['--', '-'] if len(eps_values) <= 2 else ['--', '-.', '-']
            for run_dir, session_dir in found:
                epsilon = epsilon_of(run_dir.name)
                if epsilon is None:
                    short, style = run_short_name(attack, run_dir, session_dir), ':'
                else:
                    short = f'{attack.upper()} ε{epsilon:g}'
                    style = budget_styles[min(eps_values.index(epsilon),
                                              len(budget_styles) - 1)]
                conditions.append((
                    f'{attack}-{run_dir.name}', short,
                    attack_label(attack, run_dir, session_dir),
                    make_adv_dataset(clean_dataset, session_dir),
                    ATTACK_COLORS.get(attack, '#777777'),
                    style, ATTACK_MARKERS.get(attack, 'o'), str(run_dir)))

        series = []
        for condition, short, label, dataset, color, style, marker, run_ref in conditions:
            if condition == 'clean':
                item = {
                    'condition': condition,
                    'short': short,
                    'label': label,
                    'run': run_ref,
                    'color': color,
                    'style': style,
                    'marker': marker,
                    'images': len(clean_dataset),
                    'scale': clean_scale.copy(),
                    'long': clean_long.copy(),
                    'identity': clean_identity.copy() if clean_identity is not None else None,
                    'transition': None,
                    'counts': None,
                    'per_image': None,
                }
            else:
                print(f'  分析 {short} ...')
                result = paired_condition(
                    clean_dataset, dataset, clean_hist if clean_hist is not None else None,
                    bank,
                    {
                        'batch_size': args.batch_size,
                        'num_workers': args.num_workers,
                        'filter_chunk': args.filter_chunk,
                        'dominance': args.dominance,
                        'code_grid': args.code_grid,
                        'per_image': args.per_image,
                    },
                    device, desc=f'{name}-{condition}')
                identity = None
                if not args.no_identity:
                    adv_hist_device = result['adv_hist'].to(device)
                    identity, _ = identity_rows(
                        clean_hist_device, adv_hist_device, clean_labels_device,
                        gallery_descriptors, gallery_labels_device, features,
                        len(gallery_dataset), bank, clean_rank1=clean_rank1)
                item = {
                    'condition': condition,
                    'short': short,
                    'label': label,
                    'run': run_ref,
                    'color': color,
                    'style': style,
                    'marker': marker,
                    'images': result['images'],
                    'scale': result['scale'],
                    'long': result['long'],
                    'identity': identity,
                    'transition': result['transition'],
                    'counts': result['counts'],
                    'per_image': result['per_image'],
                }
            series.append(item)

            robustness_frames.append(tag_frame(
                item['long'], name, architecture, condition, run_ref,
                args.dominance, session))
            scale_frames.append(tag_frame(
                item['scale'], name, architecture, condition, run_ref,
                args.dominance, session))
            if item['identity'] is not None:
                identity_frames.append(tag_frame(
                    item['identity'], name, architecture, condition, run_ref,
                    args.dominance, session))
            if item['transition'] is not None:
                transition_frames.append(transition_frame(
                    [item], bank, name, architecture, args.dominance, session))
            if item['per_image'] is not None:
                per_image_frames.append(tag_frame(
                    item['per_image'], name, architecture, condition, run_ref,
                    args.dominance, session))

        print_condition_table(name, series)
        print_scale_table(name, series, bank)
        print_orientation_table(name, series, bank)
        if not args.no_identity:
            print_identity_table(name, series, bank)

        if not args.no_save:
            prefix = output_dir / f'session{session}_{args.dominance}_{stamp}_{name}'
            saved = plot_heatmaps(
                name, series, bank, 'code_retention_pooled',
                '方向码保留率 P(对抗码 = 干净码)', '码保留率',
                prefix.with_name(prefix.name + '_retention'), 0.0, 1.0)
            if saved is not None:
                print(f'  已保存：{saved}')
            saved = plot_heatmaps(
                name, series, bank, 'magnitude_corr_mean',
                '干净/对抗方向响应相关性', 'Pearson 相关',
                prefix.with_name(prefix.name + '_corr'), None, 1.0)
            if saved is not None:
                print(f'  已保存：{saved}')
            saved = plot_curves(name, series, bank,
                                prefix.with_name(prefix.name + '_curves'))
            print(f'  已保存：{saved}')
            if not args.no_identity:
                saved = plot_identity(name, series, bank,
                                      prefix.with_name(prefix.name + '_identity'))
                if saved is not None:
                    print(f'  已保存：{saved}')
            saved = plot_transitions(name, series, bank,
                                     prefix.with_name(prefix.name + '_transition'))
            if saved is not None:
                print(f'  已保存：{saved}')

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if args.no_save:
        return

    prefix = output_dir / f'session{session}_{args.dominance}_{stamp}'
    paths = []
    path = prefix.with_name(prefix.name + '_robustness.csv')
    pd.concat(robustness_frames, ignore_index=True).to_csv(
        path, index=False, encoding='utf-8-sig')
    paths.append(path)
    path = prefix.with_name(prefix.name + '_scale.csv')
    pd.concat(scale_frames, ignore_index=True).to_csv(
        path, index=False, encoding='utf-8-sig')
    paths.append(path)
    if identity_frames:
        path = prefix.with_name(prefix.name + '_identity.csv')
        pd.concat(identity_frames, ignore_index=True).to_csv(
            path, index=False, encoding='utf-8-sig')
        paths.append(path)
    if transition_frames:
        path = prefix.with_name(prefix.name + '_transitions.csv')
        pd.concat(transition_frames, ignore_index=True).to_csv(
            path, index=False, encoding='utf-8-sig')
        paths.append(path)
    if per_image_frames:
        path = prefix.with_name(prefix.name + '_images.csv')
        pd.concat(per_image_frames, ignore_index=True).to_csv(
            path, index=False, encoding='utf-8-sig')
        paths.append(path)

    print('\n汇总文件：')
    for path in paths:
        print(f'  {path}')


if __name__ == '__main__':
    main()
