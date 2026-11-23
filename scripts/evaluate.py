"""评估 train_weights 中已训练掌纹模型的识别准确率。

在项目根目录运行：
    python scripts/evaluate.py                     评估默认配置中的全部模型
    python scripts/evaluate.py --model resnet18    只评估一个模型
    python scripts/evaluate.py --session 1          评估训练批次
    python scripts/evaluate.py --limit 100          只取前 100 张图快速自检
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from data.dataset import IMAGES_PER_ID, make_session_dataset  # noqa: E402
from evaluations.metrics import evaluate  # noqa: E402
from models.manager import (  # noqa: E402
    DEFAULT_CONFIG, load_config, load_pretrained, resolve_path, select_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', action='append',
                        help='模型名称，可重复；默认评估配置中的全部模型')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='模型登记与评估配置文件')
    parser.add_argument('--session', type=int, help='同济采集批次，1 或 2')
    parser.add_argument('--data-root', type=Path, help='包含 Tongji/ROI 的数据根目录')
    parser.add_argument('--device', default='auto', help='auto、cpu 或 cuda[:编号]')
    parser.add_argument('--batch-size', type=int, help='推理批量')
    parser.add_argument('--num-workers', type=int, help='DataLoader 工作进程数')
    parser.add_argument('--limit', type=int, help='只评估前 N 张图，便于快速自检')
    parser.add_argument('--output', type=Path, help='结果 CSV 输出目录')
    parser.add_argument('--no-save', action='store_true', help='只打印结果，不写 CSV')
    return parser.parse_args()


def results_frame(results):
    """把嵌套结果摊平成一行一个模型的表格。"""
    rows = []
    for item in results:
        metrics = item['metrics']
        k = metrics['top_k']
        rows.append({
            'model': item['model'],
            'description': item['description'],
            'architecture': item['architecture'],
            'num_classes': item['num_classes'],
            'weights': item['weights'],
            'checkpoint_epoch': item['checkpoint']['epoch'],
            'checkpoint_val_loss': item['checkpoint']['val_loss'],
            'checkpoint_val_acc': item['checkpoint']['val_acc'],
            'dataset': item['dataset']['name'],
            'session': item['dataset']['session'],
            'images': item['dataset']['images'],
            'images_per_id': item['dataset']['images_per_id'],
            'loss': metrics['loss'],
            'top1_pct': metrics['top1_pct'],
            f'top{k}_pct': metrics[f'top{k}_pct'],
            'correct': metrics['correct'],
            'total': metrics['total'],
            'group_top1_pct': metrics.get('group_top1_pct'),
            f'group_top{k}_pct': metrics.get(f'group_top{k}_pct'),
            'group_total': metrics.get('group_total'),
            'evaluated_at': item['evaluated_at'],
        })
    return pd.DataFrame(rows)


def print_summary(results):
    """打印各模型的识别率对比表。"""
    print('\n' + '=' * 78)
    print(f"{'模型':<12}{'单张Top-1':>12}{'单张Top-5':>12}{'融合Top-1':>13}{'融合Top-5':>13}")
    print('-' * 78)
    for item in results:
        metrics = item['metrics']
        k = metrics['top_k']
        top1 = metrics['top1_pct']
        top_k = metrics[f'top{k}_pct']
        group_top1 = metrics.get('group_top1_pct')
        group_top_k = metrics.get(f'group_top{k}_pct')
        fusion_top1 = f'{group_top1:.2f}%' if group_top1 is not None else '-'
        fusion_top_k = f'{group_top_k:.2f}%' if group_top_k is not None else '-'
        print(f"{item['model']:<12}{top1:>11.2f}%{top_k:>11.2f}%"
              f"{fusion_top1:>13}{fusion_top_k:>13}")
    print('=' * 78)


def main():
    args = parse_args()
    config = load_config(args.config)
    settings = config['evaluation']

    session = args.session if args.session is not None else settings.get('session', 2)
    data_root = args.data_root or resolve_path(settings.get('data_root', 'data/resources'))
    batch_size = args.batch_size or settings.get('batch_size', 32)
    num_workers = args.num_workers if args.num_workers is not None else settings.get('num_workers', 0)
    output_dir = args.output or resolve_path(settings.get('output_dir', 'results/evaluate'))
    names = args.model or list(config['models'])
    device = select_device(args.device)

    unknown = [name for name in names if name not in config['models']]
    if unknown:
        raise SystemExit(f'配置中没有模型 {", ".join(unknown)}；'f'可选 {", ".join(config["models"])}')

    print(f'设备：{device}，数据：{data_root}（session{session}）')
    dataset = make_session_dataset(session, data_root, limit=args.limit)
    print(f'评估图像：{len(dataset)} 张')
    if args.limit and args.limit % IMAGES_PER_ID:
        print(f'提示：--limit 不是 {IMAGES_PER_ID} 的整数倍，' f'融合识别率只统计凑满 {IMAGES_PER_ID} 张图的手掌')

    results = []
    for name in names:
        model, info = load_pretrained(name, device=device, config=config)
        print(f"\n[{name}] {info['architecture']}，{info['num_classes']} 类，权重 {info['weights']}")
        if info.get('epoch') is not None:
            print(f"        checkpoint epoch {info['epoch']}，" f"训练记录验证 Top-1 {info['val_acc']:.2f}%")

        metrics = evaluate(model, dataset, device, batch_size=batch_size,
                           images_per_id=IMAGES_PER_ID, num_workers=num_workers,
                           desc=name, name=name)
        results.append({
            'model': name,
            'description': info.get('description'),
            'architecture': info['architecture'],
            'num_classes': info['num_classes'],
            'weights': info['weights'],
            'checkpoint': {
                'epoch': info.get('epoch'),
                'val_loss': info.get('val_loss'),
                'val_acc': info.get('val_acc'),
            },
            'dataset': {
                'name': 'tongji',
                'session': session,
                'root': str(data_root),
                'images': len(dataset),
                'images_per_id': IMAGES_PER_ID,
            },
            'metrics': metrics,
            'evaluated_at': datetime.now().isoformat(timespec='seconds'),
        })
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print_summary(results)

    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = output_dir / f'session{session}_{stamp}.csv'
        # utf-8-sig 便于 Excel 直接打开中文和 Windows 路径。
        results_frame(results).to_csv(path, index=False, encoding='utf-8-sig')
        print(f'结果已保存：{path}')
    return results


if __name__ == '__main__':
    main()
