"""生成对抗样本并保存到 outputs/adv 下。

在项目根目录运行（--attack 省略时默认 cw，即 L∞ 版本）：
    python scripts/generate_adv.py --attack cw   --model resnet18 --epsilon 4
    python scripts/generate_adv.py --attack cw-l2 --model vgg16 --c 1 --steps 50
    python scripts/generate_adv.py --attack pgd  --model resnet18 --epsilon 4
    python scripts/generate_adv.py --attack apgd --model resnet18 --steps 30

约定：
    * --epsilon 与 --alpha 的单位都是 1/255，--epsilon 4 表示 4/255；
    * 攻击在 [0, 1] 灰度像素空间进行，保存为 PNG，便于防御评估直接复用；
    * 每个输出目录包含：
        - 对抗样本图片（与原图同名，扩展名改为 .png）
        - metadata.csv：逐图的原标签、干净预测、对抗预测、是否成功、L∞ 扰动
          （“成功”要求干净预测正确且对抗预测错误，原本分错的图一律记为 0）
        - run_info.json：本次运行的完整参数与汇总指标
        - summary.csv：一行汇总，方便多个模型横向比较
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from attacks import NormalizedModel, build_attack  # noqa: E402
from data.dataset import IMAGE_SIZE, IMAGES_PER_ID, NUM_CLASSES, PalmprintDataset  # noqa: E402
from models.manager import (  # noqa: E402
    DEFAULT_CONFIG, load_config, load_pretrained, resolve_path, select_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--attack', default='cw',
                        choices=['pgd', 'bim', 'cw', 'cw-l2', 'apgd'],
                        help='攻击方法，默认 cw（L∞ 版本）；cw-l2 为 L2 版本')
    parser.add_argument('--model', action='append',
                        help='模型名称，可重复；默认配置中的全部模型')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='模型与攻击配置文件')
    parser.add_argument('--session', type=int, help='同济采集批次，1 或 2')
    parser.add_argument('--data-root', type=Path, help='包含 Tongji/ROI 的数据根目录')
    parser.add_argument('--device', default='auto', help='auto、cpu 或 cuda[:编号]')
    parser.add_argument('--batch-size', type=int, help='生成对抗样本时的批量')
    parser.add_argument('--num-workers', type=int, help='DataLoader 工作进程数')
    parser.add_argument('--limit', type=int, help='只处理前 N 张图，便于快速自检')
    parser.add_argument('--epsilon', type=float,
                        help='L∞ 扰动上限，单位 1/255（默认读取配置，4 表示 4/255）')
    parser.add_argument('--steps', type=int, help='迭代步数')
    parser.add_argument('--alpha', type=float,
                        help='每步步长，单位 1/255；默认按 epsilon/steps*alpha_scale 自动计算')
    parser.add_argument('--alpha-scale', type=float, help='自动步长倍率')
    parser.add_argument('--kappa', type=float, help='CW 置信度边界')
    parser.add_argument('--c', type=float, help='CW-L2 边界损失权重')
    parser.add_argument('--lr', type=float, help='CW-L2 的 Adam 学习率')
    parser.add_argument('--rho', type=float, help='APGD 振荡判据')
    parser.add_argument('--random-start', action=argparse.BooleanOptionalAction,
                        default=None, help='PGD / CW-L∞ 是否随机初始化（默认读取配置）')
    parser.add_argument('--seed', type=int, help='随机种子，默认读取配置')
    parser.add_argument('--output', type=Path, help='对抗样本输出根目录')
    return parser.parse_args()


def pick(cli_value, config, key, default):
    """命令行参数优先，其次配置文件，最后使用代码内默认值。"""
    return cli_value if cli_value is not None else config.get(key, default)


def make_pixel_dataset(session, root, limit=None):
    """构建像素空间数据集：输出 [1, 224, 224]、取值 [0, 1] 的灰度图。

    与 data.dataset.make_session_dataset 使用同一套文件编号和标签，
    但不做 ImageNet 归一化，攻击在该像素空间内叠加扰动。
    """
    if session not in (1, 2):
        raise ValueError('采集批次必须为 1 或 2')
    if limit is not None and limit <= 0:
        raise ValueError('limit 必须为正整数')

    directory = Path(root).resolve() / 'Tongji' / 'ROI' / f'session{session}'
    total = NUM_CLASSES * IMAGES_PER_ID
    if limit is not None:
        total = min(total, limit)

    samples = []
    for i in range(1, total + 1):
        path = directory / f'{i:05d}.bmp'
        if not path.is_file():
            raise FileNotFoundError(f'{directory} 缺少 {path.name}，请检查数据目录')
        samples.append((path, (i - 1) // IMAGES_PER_ID))

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),  # 灰度图变为 [1, H, W]，自动缩放到 0~1
    ])
    return PalmprintDataset(samples, transform)


def build_params(args, attack_config):
    """把命令行覆盖项与配置默认值合并成攻击构造参数。"""
    # L2 版本不使用 epsilon，单独处理。
    if args.attack == 'cw-l2':
        config = attack_config.get('cw_l2', {})
        return {
            'c': float(pick(args.c, config, 'c', 1.0)),
            'kappa': float(pick(args.kappa, config, 'kappa', 0.0)),
            'steps': int(pick(args.steps, config, 'steps', 50)),
            'lr': float(pick(args.lr, config, 'lr', 0.01)),
        }

    epsilon = float(pick(args.epsilon, attack_config, 'epsilon', 4)) / 255.0
    alpha = args.alpha / 255.0 if args.alpha is not None else None

    if args.attack == 'cw':
        config = attack_config.get('cw', {})
        return {
            'epsilon': epsilon,
            'steps': int(pick(args.steps, config, 'steps', 50)),
            'alpha': alpha,
            'alpha_scale': float(pick(args.alpha_scale, config, 'alpha_scale', 2.5)),
            'kappa': float(pick(args.kappa, config, 'kappa', 0.0)),
            'random_start': bool(pick(args.random_start, config, 'random_start', True)),
        }

    if args.attack == 'pgd':
        config = attack_config.get('pgd', {})
        return {
            'epsilon': epsilon,
            'steps': int(pick(args.steps, config, 'steps', 10)),
            'alpha': alpha,
            'alpha_scale': float(pick(args.alpha_scale, config, 'alpha_scale', 2.5)),
            'random_start': bool(pick(args.random_start, config, 'random_start', True)),
        }

    if args.attack == 'bim':
        config = attack_config.get('bim', {})
        return {
            'epsilon': epsilon,
            'steps': int(pick(args.steps, config, 'steps', 10)),
            'alpha': alpha,
            'alpha_scale': float(pick(args.alpha_scale, config, 'alpha_scale', 1.0)),
        }

    config = attack_config.get('apgd', {})
    return {
        'epsilon': epsilon,
        'steps': int(pick(args.steps, config, 'steps', 30)),
        'rho': float(pick(args.rho, config, 'rho', 0.75)),
    }


def strength_name(attack, params):
    """用攻击强度参数命名子目录，便于不同设置的结果并存。"""
    if attack == 'cw-l2':
        return f"c{params['c']:g}_kappa{params['kappa']:g}"
    # L∞ 攻击统一用 epsN 命名，便于分析脚本按目录名解析 epsilon；
    # kappa 等其余参数记录在 run_info.json 中。
    return f"eps{params['epsilon'] * 255:g}"


def save_batch(adv, images, labels, clean_pred, adv_pred, dataset, offset,
               save_dir, rows):
    """保存一个批次的对抗图片并记录逐图指标。"""
    linf = (adv - images).abs().flatten(1).amax(dim=1)
    for index in range(adv.shape[0]):
        source = dataset.samples[offset + index][0]
        filename = source.stem + '.png'
        TF.to_pil_image(adv[index].cpu()).save(save_dir / filename)
        rows.append({
            'filename': filename,
            'source': str(source),
            'label': int(labels[index].item()),
            'clean_pred': int(clean_pred[index].item()),
            'adv_pred': int(adv_pred[index].item()),
            'success': int(clean_pred[index].item() == labels[index].item()
                           and adv_pred[index].item() != labels[index].item()),
            'linf': f'{linf[index].item():.8f}',
        })


def run_one(model_name, model, info, attack_name, params, dataset, loader,
            save_dir, device):
    """对一个模型执行指定攻击，保存样本并返回汇总指标。"""
    wrapped = NormalizedModel(model).to(device)
    attack = build_attack(attack_name, wrapped, **params)
    # 记录攻击实例里的实际参数（例如自动计算出的 alpha），而不是命令行原值。
    effective_params = {
        key: value for key, value in vars(attack).items() if key != 'model'}

    save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total = clean_correct = adv_correct = success = 0
    linf_sum = 0.0
    seen = 0

    for images, labels in tqdm(loader, desc=f'{model_name}/{attack_name}',
                               leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # 先攻击，再量化到 8 位，使指标与最终保存的 PNG 完全一致。
        adv = attack.attack(images, labels)
        adv = torch.round(adv.clamp(0.0, 1.0) * 255.0) / 255.0

        with torch.no_grad():
            clean_pred = wrapped(images).argmax(dim=1)
            adv_pred = wrapped(adv).argmax(dim=1)

        batch = images.shape[0]
        total += batch
        clean_correct += (clean_pred == labels).sum().item()
        adv_correct += (adv_pred == labels).sum().item()
        success += ((clean_pred == labels) & (adv_pred != labels)).sum().item()
        linf_sum += (adv - images).abs().flatten(1).amax(dim=1).sum().item()

        save_batch(adv, images, labels, clean_pred, adv_pred, dataset,
                   seen, save_dir, rows)
        seen += batch

    # 两种口径分开记录，避免把“原本就分错”的样本混进攻击成功里：
    #   attack_success_pct    条件攻击成功率：分母为干净预测正确的样本，
    #                         分子要求干净预测正确且对抗预测错误；
    #   all_misclassified_pct 全样本对抗误判率：分母为全部样本，含原本分错的图。
    metrics = {
        'images': total,
        'clean_correct': clean_correct,
        'clean_wrong': total - clean_correct,
        'adv_correct': adv_correct,
        'adv_wrong': total - adv_correct,
        'success': success,
        'clean_top1_pct': 100.0 * clean_correct / total if total else 0.0,
        'adv_top1_pct': 100.0 * adv_correct / total if total else 0.0,
        'attack_success_pct': 100.0 * success / clean_correct if clean_correct else 0.0,
        'all_misclassified_pct': 100.0 * (total - adv_correct) / total if total else 0.0,
        'mean_linf': linf_sum / total if total else 0.0,
    }
    created_at = datetime.now().isoformat(timespec='seconds')

    # 逐图记录，方便后续做单张、同一手掌融合或防御评估。
    with (save_dir / 'metadata.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'filename', 'source', 'label', 'clean_pred', 'adv_pred', 'success', 'linf'])
        writer.writeheader()
        writer.writerows(rows)

    run_info = {
        'attack': attack_name,
        'model': model_name,
        'model_architecture': info.get('architecture'),
        'weights': info.get('weights'),
        'params': effective_params,
        'device': str(device),
        'session': dataset.samples[0][0].parent.name,
        'metrics': metrics,
        'created_at': created_at,
    }
    with (save_dir / 'run_info.json').open('w', encoding='utf-8') as file:
        json.dump(run_info, file, ensure_ascii=False, indent=2)

    summary = {
        'attack': attack_name,
        'model': model_name,
        'session': run_info['session'],
        **metrics,
        'params': json.dumps(effective_params, ensure_ascii=False),
        'device': str(device),
        'created_at': created_at,
    }
    with (save_dir / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print(f'  输出目录：{save_dir}')
    return summary


def print_summary(rows):
    """打印本次运行各模型的干净/对抗识别率对比。"""
    print('\n' + '=' * 92)
    print(f"{'模型':<12}{'攻击':<8}{'干净Top-1':>12}{'对抗Top-1':>12}"
          f"{'条件ASR':>11}{'全样本误判':>12}{'平均L∞':>12}")
    print('-' * 92)
    for row in rows:
        print(f"{row['model']:<12}{row['attack']:<8}"
              f"{row['clean_top1_pct']:>11.2f}%{row['adv_top1_pct']:>11.2f}%"
              f"{row['attack_success_pct']:>10.2f}%{row['all_misclassified_pct']:>11.2f}%"
              f"{row['mean_linf']:>12.5f}")
    print('=' * 92)
    print('条件ASR = 干净预测正确且对抗预测错误 / 干净预测正确（不含原本分错的图）；')
    print('全样本误判 = 对抗预测错误 / 全部样本（含原本分错的图）；'
          '对抗Top-1 也是按全部样本统计。')


def main():
    args = parse_args()
    config = load_config(args.config)
    settings = config['evaluation']
    attack_config = config.get('attacks') or {}

    session = args.session if args.session is not None else settings.get('session', 2)
    data_root = args.data_root or resolve_path(settings.get('data_root', 'data/resources'))
    batch_size = args.batch_size or settings.get('batch_size', 32)
    num_workers = (args.num_workers if args.num_workers is not None
                   else settings.get('num_workers', 0))
    output_root = (args.output or resolve_path(
        attack_config.get('output_dir', 'outputs/adv')))
    seed = int(pick(args.seed, attack_config, 'seed', 42))
    device = select_device(args.device)

    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    names = args.model or list(config['models'])
    unknown = [name for name in names if name not in config['models']]
    if unknown:
        raise SystemExit(f'配置中没有模型 {", ".join(unknown)}；'
                         f'可选 {", ".join(config["models"])}')

    params = build_params(args, attack_config)
    strength = strength_name(args.attack, params)

    print(f'设备：{device}，攻击：{args.attack}，强度：{strength}，'
          f'数据：{data_root}（session{session}）')
    print(f'参数：{params}')

    dataset = make_pixel_dataset(session, data_root, limit=args.limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=device.type == 'cuda')
    print(f'处理图像：{len(dataset)} 张')

    summaries = []
    for name in names:
        model, info = load_pretrained(name, device=device, config=config)
        print(f"\n[{name}] {info['architecture']}，{info['num_classes']} 类，"
              f"权重 {info['weights']}")
        save_dir = output_root / args.attack / name / strength / f'session{session}'
        summaries.append(run_one(name, model, info, args.attack, params,
                                 dataset, loader, save_dir, device))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print_summary(summaries)
    return summaries


if __name__ == '__main__':
    main()
