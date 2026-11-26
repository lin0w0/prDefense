"""LOTA 位平面噪声分析：对比干净样本与对抗样本的低位平面统计。

在项目根目录运行：
    python scripts/analyze_bitplane_noise.py
    python scripts/analyze_bitplane_noise.py --limit 1200 --balance-classes
    python scripts/analyze_bitplane_noise.py --epsilon 4 --attack pgd
    python scripts/analyze_bitplane_noise.py --bits 1 --patch-size 16
    python scripts/analyze_bitplane_noise.py --patch-mode random --num-patches 49
    python scripts/analyze_bitplane_noise.py --export-dir outputs/bitplane --export-limit 16
    python scripts/analyze_bitplane_noise.py --no-save

流程：
    1. 读取与评估相同文件编号的干净图像，以及 outputs/adv 下各攻击的对抗样本；
       只做图像统计，不做模型前向，因此不受模型设备影响；
    2. 取每张图低 k 位平面并拉伸（LOTA 的 bit_mode='scaling'）；
    3. 按 patch_size 切块，用 LOTA 的局部差分能量（水平/垂直/两条对角线
       相邻像素绝对差之和）给每块打分；
    4. 统计整图低位噪声强度、块分数分布，并按 patch_mode 选出代表块；
    5. 汇总干净与各攻击各预算的分布，输出 CSV、箱线图和位平面能量曲线，
       可选地把选中的噪声块放大后另存为图片。

可用 --epsilon 只保留某个 L∞ 预算；CW 等不以 eps 命名预算的运行会直接使用。
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from data.dataset import (  # noqa: E402
    IMAGE_SIZE,
    PalmprintDataset,
    make_session_dataset,
)
from defense.bitplane import (  # noqa: E402
    BIT_MODES,
    PATCH_MODES,
    bit_plane_energy,
    extract_patches,
    low_bit_plane,
    patch_score,
    select_patch_index,
)
from models.manager import DEFAULT_CONFIG, load_config, resolve_path  # noqa: E402

DEFAULT_ATTACKS = ("pgd", "bim", "cw", "apgd")
CLEAN_COLOR = "#333333"
ATTACK_COLORS = {
    "pgd": "#D55E00",
    "bim": "#0072B2",
    "cw": "#009E73",
    "apgd": "#CC79A7",
}
# 汇总表里逐条件统计的指标。
METRICS = (
    "noise_mean",
    "noise_std",
    "patch_score_max",
    "patch_score_min",
    "patch_score_mean",
    "patch_score_std",
    "selected_score",
)
PIXEL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),  # 灰度图 -> [1, H, W]，取值 [0, 1]
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", action="append", help="模型名称，可重复；默认分析配置中的全部模型"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="模型登记与数据集配置文件"
    )
    parser.add_argument("--session", type=int, help="同济采集批次，1 或 2")
    parser.add_argument("--data-root", type=Path, help="包含 Tongji/ROI 的数据根目录")
    parser.add_argument(
        "--adv-root", type=Path, help="对抗样本根目录，默认读取 attacks.output_dir"
    )
    parser.add_argument(
        "--attack",
        action="append",
        choices=DEFAULT_ATTACKS,
        help="要分析的攻击，可重复；默认全部",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="只保留某个 epsN 预算，单位 1/255；默认分析全部预算",
    )
    parser.add_argument(
        "--limit", type=int, help="干净与对抗样本都只取 N 张，便于快速自检"
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="配合 --limit 按类别均匀抽样，覆盖全部 600 类",
    )
    parser.add_argument(
        "--bits", type=int, default=3, help="取最低的 k 位平面，LOTA 默认 3"
    )
    parser.add_argument(
        "--bit-mode",
        default="scaling",
        choices=BIT_MODES,
        help="低位平面处理方式，默认 scaling（与 LOTA 一致）",
    )
    parser.add_argument(
        "--patch-size", type=int, default=32, help="切块边长，LOTA 默认 32"
    )
    parser.add_argument(
        "--patch-mode",
        default="max",
        choices=PATCH_MODES,
        help="代表块选择规则，默认 max（差分能量最大）",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        help="设置后按 LOTA 方式随机裁这么多块；默认非重叠切块",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机裁块/随机选块的种子")
    parser.add_argument("--batch-size", type=int, default=64, help="统计时的批量")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader 工作进程数"
    )
    parser.add_argument("--output", type=Path, help="结果输出目录")
    parser.add_argument("--per-image", action="store_true", help="额外保存逐图统计 CSV")
    parser.add_argument(
        "--export-dir", type=Path, help="把选中的噪声块放大后另存到该目录，便于直接查看"
    )
    parser.add_argument(
        "--export-limit",
        type=int,
        default=24,
        help="每个条件最多导出多少张噪声块，默认 24",
    )
    parser.add_argument("--no-save", action="store_true", help="只打印结果，不写文件")
    return parser.parse_args()


def balanced_subset(samples, limit):
    """从完整清单中均匀抽取 limit 张，保证每个类别都有样本。"""
    by_class = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)
    classes = sorted(by_class)
    base, extra = divmod(limit, len(classes))
    if base < 1:
        raise ValueError(f"limit={limit} 小于类别数 {len(classes)}，无法覆盖全部类别")

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
        adv_path = session_dir / f"{path.stem}.png"
        if not adv_path.is_file():
            raise FileNotFoundError(f"{session_dir} 缺少 {adv_path.name}")
        samples.append((adv_path, label))
    return PalmprintDataset(samples, PIXEL_TRANSFORM)


def epsilon_of(name):
    """从 eps1 / eps4 这样的目录名解析 epsilon，其他名称返回 None。"""
    match = re.fullmatch(r"eps(\d+(?:\.\d+)?)", name)
    return float(match.group(1)) if match else None


def find_attack_runs(adv_root, model_name, attack, session, epsilon=None):
    """定位一次攻击的全部运行目录，返回 [(运行目录, session 子目录), ...]。"""
    attack_dir = Path(adv_root) / attack / model_name
    if not attack_dir.is_dir():
        return []
    runs = sorted(
        directory
        for directory in attack_dir.iterdir()
        if directory.is_dir() and (directory / f"session{session}").is_dir()
    )
    if not runs:
        return []

    eps_runs = [
        directory for directory in runs if epsilon_of(directory.name) is not None
    ]
    if eps_runs:
        if epsilon is not None:
            eps_runs = [
                directory
                for directory in eps_runs
                if epsilon_of(directory.name) == float(epsilon)
            ]
        eps_runs.sort(key=lambda directory: epsilon_of(directory.name))
        return [(run, run / f"session{session}") for run in eps_runs]

    if epsilon is not None:
        return []
    if len(runs) > 1:
        names = "、".join(directory.name for directory in runs)
        raise ValueError(f"{attack_dir} 有多个非 epsilon 运行（{names}），请手动清理")
    return [(runs[0], runs[0] / f"session{session}")]


def attack_label(attack, run_dir, session_dir):
    """图例名称：L∞ 攻击标注 epsilon，CW 标注 c。"""
    eps = epsilon_of(run_dir.name)
    if eps is not None:
        return f"{attack.upper()}（ε={eps:g}/255）"
    try:
        with (Path(session_dir) / "run_info.json").open(encoding="utf-8") as file:
            params = json.load(file).get("params", {})
    except (OSError, ValueError):
        params = {}
    if attack == "cw" and "c" in params:
        return f"CW（c={params['c']:g}）"
    return attack.upper()


def collect_condition(dataset, settings, export_dir=None, export_limit=0, desc=""):
    """逐图统计低位平面噪声，返回包含 filename/label 和各项指标的 DataFrame。"""
    loader = DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
    )
    frames = []
    seen = exported = 0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        noise = low_bit_plane(images, settings["bits"], settings["bit_mode"])
        patches = extract_patches(
            noise,
            settings["patch_size"],
            settings["num_patches"],
            settings["generator"],
        )
        scores = patch_score(patches)
        choice = select_patch_index(
            scores, settings["patch_mode"], settings["generator"]
        )
        batch = images.shape[0]
        row_index = torch.arange(batch)
        selected = patches[row_index, choice]
        energy = bit_plane_energy(images, 8)

        noise_flat = noise.flatten(1)
        frame = pd.DataFrame(
            {
                "filename": [
                    Path(path).name for path, _ in dataset.samples[seen : seen + batch]
                ],
                "label": labels.numpy(),
                "noise_mean": noise_flat.mean(dim=1).numpy(),
                "noise_std": noise_flat.std(dim=1, unbiased=False).numpy(),
                "patch_score_max": scores.max(dim=1).values.numpy(),
                "patch_score_min": scores.min(dim=1).values.numpy(),
                "patch_score_mean": scores.mean(dim=1).numpy(),
                "patch_score_std": scores.std(dim=1, unbiased=False).numpy(),
                "selected_score": scores[row_index, choice].numpy(),
            }
        )
        for bit in range(8):
            frame[f"bit{bit + 1}_energy"] = energy[:, bit].numpy()
        frames.append(frame)

        if export_dir is not None and exported < export_limit:
            count = min(export_limit - exported, batch)
            upscaled = F.interpolate(
                selected[:count],
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for index in range(count):
                name = Path(dataset.samples[seen + index][0]).stem + ".png"
                TF.to_pil_image(upscaled[index].clamp(0, 1).cpu()).save(
                    export_dir / name
                )
            exported += count
        seen += batch

    return pd.concat(frames, ignore_index=True)


def summarize_condition(frame, model, architecture, condition, run):
    """把一个条件内的逐图指标汇总成长表：一行一个指标。"""
    rows = []
    for metric in METRICS:
        values = frame[metric]
        rows.append(
            {
                "model": model,
                "architecture": architecture,
                "condition": condition,
                "run": run,
                "metric": metric,
                "images": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=0),
                "min": values.min(),
                "p05": values.quantile(0.05),
                "median": values.median(),
                "p95": values.quantile(0.95),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def print_table(model_name, series):
    """打印各条件在几个关键指标上的均值。"""
    print(f"\n[{model_name}] 低位平面噪声统计")
    header = (
        f"{'条件':<16}{'样本':>7}{'噪声均值':>12}"
        f"{'块分数均值':>14}{'最强块':>14}{'选中块':>14}"
    )
    print(header)
    print("-" * len(header))
    for _, short, frame, *_ in series:
        print(
            f"{short:<16}{len(frame):>7}"
            f"{frame['noise_mean'].mean():>12.4f}"
            f"{frame['patch_score_mean'].mean():>14.2f}"
            f"{frame['patch_score_max'].mean():>14.2f}"
            f"{frame['selected_score'].mean():>14.2f}"
        )


def plot_model(model_name, series, bits, output_prefix):
    """左图：最强噪声块的差分能量箱线图；右图：位平面置 1 比例相对干净样本的变化。"""
    # Windows 自带的微软雅黑/黑体可以正常显示中文坐标轴。
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, (left, right) = plt.subplots(1, 2, figsize=(14.0, 5.6))

    boxes = left.boxplot(
        [frame["patch_score_max"].to_numpy() for _, _, frame, *_ in series],
        patch_artist=True,
        showfliers=False,
        widths=0.62,
    )
    for patch, (_, _, _, color, _) in zip(boxes["boxes"], series):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.35)
    for median in boxes["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.4)
    left.set_xticks(range(1, len(series) + 1))
    left.set_xticklabels(
        [label for label, *_ in series], rotation=30, ha="right", fontsize=8
    )
    left.set_ylabel("最强块的差分能量（patch_score_max）")
    left.set_title("每张图最强噪声块的差分能量分布")
    left.grid(axis="y", alpha=0.3)

    # 各位平面的置 1 比例本身由图像内容决定，画绝对值时各条件几乎重合；
    # 改为相对干净样本的变化（百分点），对抗扰动落在哪一位就一目了然。
    clean_energy = [series[0][2][f"bit{bit}_energy"].mean() for bit in range(1, 9)]
    for label, _, frame, color, style in series[1:]:
        values = [
            (frame[f"bit{bit}_energy"].mean() - clean_energy[bit - 1]) * 100.0
            for bit in range(1, 9)
        ]
        right.plot(
            range(1, 9),
            values,
            marker="o",
            markersize=3.5,
            color=color,
            linestyle=style,
            linewidth=1.8,
            label=label,
        )
    right.axhline(
        0.0, color=CLEAN_COLOR, linestyle=":", linewidth=1.2, label="干净样本（基准）"
    )
    right.set_xticks(range(1, 9))
    right.set_xlabel("位平面（1 = 最低位 LSB）")
    right.set_ylabel("相对干净样本的置 1 比例变化（百分点）")
    right.set_title("对抗样本的位平面置 1 比例偏移")
    right.grid(alpha=0.3)
    right.legend(fontsize=7.5, ncol=2, loc="best")

    figure.suptitle(
        f"{model_name}：干净样本与对抗样本的 LOTA 低位平面噪声（低 {bits} 位）"
    )
    figure.tight_layout()

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main():
    args = parse_args()
    if args.no_save and args.export_dir is not None:
        raise SystemExit("--no-save 与 --export-dir 冲突：前者不写任何文件")
    if args.export_limit < 0:
        raise SystemExit("--export-limit 不能为负数")
    if args.num_patches is not None and args.num_patches < 1:
        raise SystemExit("--num-patches 必须为正整数")

    config = load_config(args.config)
    settings = config["evaluation"]
    attack_settings = config.get("attacks", {})

    session = args.session if args.session is not None else settings.get("session", 2)
    data_root = args.data_root or resolve_path(
        settings.get("data_root", "data/resources")
    )
    adv_root = args.adv_root or resolve_path(
        attack_settings.get("output_dir", "outputs/adv")
    )
    output_dir = args.output or resolve_path("results/bitplane_noise")
    names = args.model or list(config["models"])
    attacks = args.attack or list(DEFAULT_ATTACKS)

    unknown = [name for name in names if name not in config["models"]]
    if unknown:
        raise SystemExit(
            f"配置中没有模型 {', '.join(unknown)}；可选 {', '.join(config['models'])}"
        )

    torch.manual_seed(args.seed)
    print(
        f"数据：{data_root}（session{session}），低 {args.bits} 位平面，"
        f"{args.patch_size}×{args.patch_size} 块，选块规则：{args.patch_mode}"
    )
    clean_dataset = make_pixel_dataset(
        session, data_root, limit=args.limit, balance_classes=args.balance_classes
    )
    covered = len({label for _, label in clean_dataset.samples})
    print(f"干净样本：{len(clean_dataset)} 张，覆盖 {covered} 类")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
    if args.export_dir is not None:
        args.export_dir = args.export_dir.resolve()

    summaries = []
    per_image_frames = []
    for name in names:
        entry = config["models"][name]
        architecture = entry.get("arch")
        print(f"\n[{name}] {architecture}")

        # (条件键, 短名, 图例, 数据集, 颜色, 线型, 运行目录)
        conditions = [
            ("clean", "干净", "干净样本", clean_dataset, CLEAN_COLOR, "-", "clean")
        ]
        for attack in attacks:
            found = find_attack_runs(adv_root, name, attack, session, args.epsilon)
            if not found:
                detail = f"eps{args.epsilon:g}" if args.epsilon is not None else "可用"
                print(
                    f"  跳过 {attack}：{adv_root / attack / name} 没有 {detail} 的 "
                    f"session{session} 运行"
                )
                continue
            eps_values = sorted(
                epsilon_of(run.name)
                for run, _ in found
                if epsilon_of(run.name) is not None
            )
            budget_styles = ["--", "-"] if len(eps_values) <= 2 else ["--", "-.", "-"]
            for run_dir, session_dir in found:
                epsilon = epsilon_of(run_dir.name)
                if epsilon is None:
                    short, style = attack.upper(), "-"
                else:
                    short = f"{attack.upper()} ε{epsilon:g}"
                    style = budget_styles[
                        min(eps_values.index(epsilon), len(budget_styles) - 1)
                    ]
                conditions.append(
                    (
                        f"{attack}-{run_dir.name}",
                        short,
                        attack_label(attack, run_dir, session_dir),
                        make_adv_dataset(clean_dataset, session_dir),
                        ATTACK_COLORS.get(attack, "#777777"),
                        style,
                        str(run_dir),
                    )
                )

        series = []
        for condition, short, label, dataset, color, style, run_ref in conditions:
            # 每个条件单独重置随机数，让各条件的随机裁块/选块互相可比。
            generator = torch.Generator().manual_seed(args.seed)
            export_dir = None
            if args.export_dir is not None:
                export_dir = args.export_dir / name / condition
                export_dir.mkdir(parents=True, exist_ok=True)

            frame = collect_condition(
                dataset,
                {
                    "bits": args.bits,
                    "bit_mode": args.bit_mode,
                    "patch_size": args.patch_size,
                    "patch_mode": args.patch_mode,
                    "num_patches": args.num_patches,
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "generator": generator,
                },
                export_dir=export_dir,
                export_limit=args.export_limit,
                desc=f"{name}-{condition}",
            )

            summaries.append(
                summarize_condition(frame, name, architecture, condition, run_ref)
            )
            series.append((label, short, frame, color, style))

            if args.per_image:
                detail = frame.copy()
                detail.insert(0, "condition", condition)
                detail.insert(0, "model", name)
                per_image_frames.append(detail)

        print_table(name, series)
        if not args.no_save:
            prefix = output_dir / f"session{session}_bits{args.bits}_{stamp}_{name}"
            png_path, _ = plot_model(name, series, args.bits, prefix)
            print(f"  已保存：{png_path}")

    if args.no_save:
        return

    table = pd.concat(summaries, ignore_index=True)
    table.insert(3, "session", session)
    table.insert(4, "bits", args.bits)
    table.insert(5, "bit_mode", args.bit_mode)
    table.insert(6, "patch_size", args.patch_size)
    table.insert(7, "patch_mode", args.patch_mode)
    # utf-8-sig 便于 Excel 直接打开中文和 Windows 路径。
    csv_path = output_dir / f"session{session}_bits{args.bits}_{stamp}.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n汇总 CSV：{csv_path}")

    if per_image_frames:
        detail_path = (
            output_dir / f"session{session}_bits{args.bits}_{stamp}_images.csv"
        )
        pd.concat(per_image_frames, ignore_index=True).to_csv(
            detail_path, index=False, encoding="utf-8-sig"
        )
        print(f"逐图 CSV：{detail_path}")


if __name__ == "__main__":
    main()
