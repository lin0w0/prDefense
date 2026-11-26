"""统计原图与高斯低通滤波图像的嵌入余弦相似度随带宽半径 r 的变化。

在项目根目录运行：
    python scripts/analyze_lpf_similarity.py
    python scripts/analyze_lpf_similarity.py --limit 1500 --balance-classes
    python scripts/analyze_lpf_similarity.py --model resnet18 --limit 1200
    python scripts/analyze_lpf_similarity.py --epsilon 1 --radii 2 8 16 32 64 112
    python scripts/analyze_lpf_similarity.py --attack pgd --attack cw
    python scripts/analyze_lpf_similarity.py --no-attacks

流程：
    1. 在 [0, 1] 像素空间读取干净样本与已生成的对抗样本（同一批文件编号）；
    2. 对每个半径 r 做频域高斯低通滤波，得到低频参考图像；
    3. 原图与低频图分别做 ImageNet 归一化后送入模型，提取分类层之前的
       嵌入（ResNet-18 为 512 维，VGG-16 为 4096 维）；
    4. 逐图计算两个嵌入的余弦相似度，汇总均值、标准差和分位数。

输出为每个模型一张图：干净样本加每种攻击的每个预算各一条曲线，横轴半径
由大到小。同一种攻击共用颜色和标记，预算用线型区分（eps 越大越接近实线）；
可用 --epsilon 只保留某个预算，CW 等不以 eps 命名预算的运行会直接使用。
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
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from data.dataset import (  # noqa: E402
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, PalmprintDataset, make_session_dataset,
)
from defense.features import EmbeddingExtractor, final_classifier  # noqa: E402
from defense.lowpass import (  # noqa: E402
    FILTER_TYPES, filter_spectrum, inverse_spectrum,
)
from models.manager import (  # noqa: E402
    DEFAULT_CONFIG, load_config, load_pretrained, resolve_path, select_device,
)

# 覆盖很小到接近奈奎斯特频率（224×224 图像中心到角落约 158 频点）的半径。
DEFAULT_RADII = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 112]
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
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
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
    parser.add_argument('--device', default='auto', help='auto、cpu 或 cuda[:编号]')
    parser.add_argument('--batch-size', type=int, default=64, help='推理批量')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader 工作进程数')
    parser.add_argument('--limit', type=int, help='干净与对抗样本都只取 N 张，便于快速自检')
    parser.add_argument('--balance-classes', action='store_true',
                        help='配合 --limit 按类别均匀抽样，覆盖全部 600 类')
    parser.add_argument('--radii', type=float, nargs='+', default=DEFAULT_RADII,
                        help='低通滤波的衰减带宽半径，默认覆盖 1~112 频点')
    parser.add_argument('--filter-type', default='gaussian', choices=FILTER_TYPES,
                        help='低通滤波器类型，默认高斯')
    parser.add_argument('--order', type=int, default=2, help='巴特沃斯滤波器阶数')
    parser.add_argument('--attack', action='append', choices=DEFAULT_ATTACKS,
                        help='要绘制的攻击，可重复；默认全部')
    parser.add_argument('--epsilon', type=float,
                        help='只保留某个 epsN 预算，单位 1/255；默认绘制全部预算')
    parser.add_argument('--no-attacks', action='store_true', help='只分析干净样本')
    parser.add_argument('--output', type=Path, help='结果输出目录')
    parser.add_argument('--no-save', action='store_true', help='只打印结果，不写文件')
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
    """读取与评估一致的样本清单，但输出 [0, 1] 像素空间的三通道图像。

    balance_classes=True 时按类别均匀抽取，使 --limit 覆盖全部 600 类；
    否则与评估脚本一致，直接取清单中的前 limit 张。
    """
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
    """定位一次攻击的所有运行目录，返回 [(运行目录, session 子目录), ...]。

    默认返回全部预算（eps1、eps4 等）；传入 epsilon 时只保留对应的 epsN。
    CW 等不以 eps 命名预算的攻击直接返回其运行目录。
    """
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


def normalize(images, mean, std):
    """把 [0, 1] 像素图归一化为模型输入。"""
    return (images - mean) / std


@torch.no_grad()
def collect_similarities(model, dataset, radii, device, batch_size=64,
                         num_workers=0, filter_type='gaussian', order=2, desc='分析'):
    """返回 [N, len(radii)] 的逐图余弦相似度矩阵。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    columns = [[] for _ in radii]

    with EmbeddingExtractor(model) as extractor:
        for images, _ in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            original = F.normalize(extractor(normalize(images, mean, std)), dim=1)

            # FFT 每批只算一次，所有半径共用同一份频谱。
            spectrum = torch.fft.fftshift(torch.fft.fft2(images), dim=(-2, -1))
            for index, radius in enumerate(radii):
                low = inverse_spectrum(
                    filter_spectrum(spectrum, radius, filter_type, order))
                low_features = F.normalize(
                    extractor(normalize(low, mean, std)), dim=1)
                columns[index].append((original * low_features).sum(dim=1).cpu())

    return torch.stack([torch.cat(column) for column in columns], dim=1)


def summarize(similarities, radii):
    """把逐图相似度汇总成一行一个半径的表格。"""
    rows = []
    for index, radius in enumerate(radii):
        values = similarities[:, index]
        rows.append({
            'radius': radius,
            'images': values.numel(),
            'mean': values.mean().item(),
            'std': values.std(unbiased=False).item(),
            'min': values.min().item(),
            'p05': values.quantile(0.05).item(),
            'median': values.median().item(),
            'p95': values.quantile(0.95).item(),
            'max': values.max().item(),
        })
    return pd.DataFrame(rows)


def print_table(model_name, series):
    """打印各条件在每个半径上的相似度均值。"""
    print(f'\n[{model_name}] 余弦相似度均值')
    header = f"{'r':>6}" + ''.join(f'{short:>12}' for _, short, *_ in series)
    print(header)
    print('-' * len(header))
    for index, radius in enumerate(series[0][2]['radius']):
        line = f'{radius:>6g}'
        for _, _, frame, *_ in series:
            line += f"{frame['mean'].iloc[index]:>12.4f}"
        print(line)


def plot_model(model_name, series, filter_type, output_prefix):
    """一个模型一张图；干净样本和每种攻击各一条均值曲线，阴影为 ±1 标准差。"""
    # Windows 自带的微软雅黑/黑体可以正常显示中文坐标轴。
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    figure, axis = plt.subplots(figsize=(9.0, 5.4))

    for label, short, frame, color, marker, style in series:
        is_clean = short == '干净'
        axis.plot(frame['radius'], frame['mean'], marker=marker, markersize=4.5,
                  linewidth=2.6 if is_clean else 2.0, color=color, label=label,
                  linestyle=style, dash_capstyle='butt')
        # 8 条曲线的阴影会互相遮挡，只给干净样本画 ±1 标准差。
        if is_clean:
            axis.fill_between(frame['radius'], frame['mean'] - frame['std'],
                              frame['mean'] + frame['std'], color=color,
                              alpha=0.12, linewidth=0)

    axis.set_xlabel('低通滤波衰减带宽半径 r（频点）')
    axis.set_ylabel('原图嵌入与低通嵌入的余弦相似度')
    title = f'{model_name}：干净样本与各类对抗样本的低通嵌入相似度'
    if filter_type != 'gaussian':
        title += f'（{filter_type}）'
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=9, loc='lower left', title='样本类型')
    # 横轴按半径由大到小排列，左侧为带宽最宽（最接近原图）的一端。
    axis.invert_xaxis()
    figure.tight_layout()

    png_path = output_prefix.with_suffix('.png')
    pdf_path = output_prefix.with_suffix('.pdf')
    figure.savefig(png_path, dpi=220, bbox_inches='tight')
    figure.savefig(pdf_path, bbox_inches='tight')
    plt.close(figure)
    return png_path, pdf_path


def main():
    args = parse_args()
    config = load_config(args.config)
    settings = config['evaluation']
    attack_settings = config.get('attacks', {})

    session = args.session if args.session is not None else settings.get('session', 2)
    data_root = args.data_root or resolve_path(settings.get('data_root', 'data/resources'))
    adv_root = args.adv_root or resolve_path(attack_settings.get('output_dir', 'outputs/adv'))
    output_dir = args.output or resolve_path('results/lpf_similarity')
    radii = sorted({float(radius) for radius in args.radii}, reverse=True)
    if any(radius <= 0 for radius in radii):
        raise SystemExit('--radii 必须全部为正数')
    names = args.model or list(config['models'])
    attacks = [] if args.no_attacks else (args.attack or list(DEFAULT_ATTACKS))
    device = select_device(args.device)

    unknown = [name for name in names if name not in config['models']]
    if unknown:
        raise SystemExit(f'配置中没有模型 {", ".join(unknown)}；'
                         f'可选 {", ".join(config["models"])}')

    print(f'设备：{device}，数据：{data_root}（session{session}）')
    clean_dataset = make_pixel_dataset(session, data_root, limit=args.limit,
                                       balance_classes=args.balance_classes)
    covered = len({label for _, label in clean_dataset.samples})
    print(f'干净样本：{len(clean_dataset)} 张，覆盖 {covered} 类，'
          f'低通类型：{args.filter_type}，半径：{len(radii)} 个'
          f'（{radii[-1]:g}~{radii[0]:g}）')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for name in names:
        model, info = load_pretrained(name, device=device, config=config)
        embedding_dim = final_classifier(model).in_features
        print(f"\n[{name}] {info['architecture']}，{embedding_dim} 维嵌入，"
              f"权重 {info['weights']}")

        # (条件, 短名, 图例, 数据集, 颜色, 标记, 线型, 运行目录)
        conditions = [('clean', '干净', '干净样本', clean_dataset, CLEAN_COLOR, 's',
                       '-', 'clean')]
        for attack in attacks:
            found = find_attack_runs(adv_root, name, attack, session, args.epsilon)
            if not found:
                detail = f'eps{args.epsilon:g}' if args.epsilon is not None else '可用'
                print(f'  跳过 {attack}：{adv_root / attack / name} 没有 {detail} 的 '
                      f'session{session} 运行')
                continue
            # 同一攻击共用一个颜色/标记，预算用线型区分：越强越接近实线。
            eps_values = sorted(epsilon_of(run.name) for run, _ in found
                                if epsilon_of(run.name) is not None)
            budget_styles = ['--', '-'] if len(eps_values) <= 2 else ['--', '-.', '-']
            for run_dir, session_dir in found:
                epsilon = epsilon_of(run_dir.name)
                if epsilon is None:
                    short, style = attack.upper(), '-'
                else:
                    rank = eps_values.index(epsilon)
                    short = f'{attack.upper()} ε{epsilon:g}'
                    style = budget_styles[min(rank, len(budget_styles) - 1)]
                conditions.append((
                    f'{attack}-{run_dir.name}', short,
                    attack_label(attack, run_dir, session_dir),
                    make_adv_dataset(clean_dataset, session_dir),
                    ATTACK_COLORS.get(attack, '#777777'),
                    ATTACK_MARKERS.get(attack, 'o'), style, str(run_dir)))

        series = []
        for condition, short, label, dataset, color, marker, style, run_ref in conditions:
            similarities = collect_similarities(
                model, dataset, radii, device, batch_size=args.batch_size,
                num_workers=args.num_workers, filter_type=args.filter_type,
                order=args.order, desc=f'{name}-{condition}')
            frame = summarize(similarities, radii)
            frame.insert(0, 'model', name)
            frame.insert(1, 'architecture', info['architecture'])
            frame.insert(2, 'embedding_dim', embedding_dim)
            frame.insert(3, 'condition', condition)
            frame.insert(4, 'run', run_ref)
            series.append((label, short, frame, color, marker, style))
            frames.append(frame)

        print_table(name, series)
        if not args.no_save:
            prefix = output_dir / f'session{session}_{args.filter_type}_{stamp}_{name}'
            png_path, pdf_path = plot_model(name, series, args.filter_type, prefix)
            print(f'  已保存：{png_path}')

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if args.no_save:
        return

    table = pd.concat(frames, ignore_index=True)
    table.insert(5, 'session', session)
    table.insert(6, 'filter_type', args.filter_type)
    table.insert(7, 'filter_order', args.order)
    # utf-8-sig 便于 Excel 直接打开中文和 Windows 路径。
    csv_path = output_dir / f'session{session}_{args.filter_type}_{stamp}.csv'
    table.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n汇总 CSV：{csv_path}')


if __name__ == '__main__':
    main()
