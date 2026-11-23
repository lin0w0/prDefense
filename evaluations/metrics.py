"""单张与同一手掌多图融合的识别率，以及可供对抗攻击复用的 logits 采集。"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.inference_mode()
def collect_logits(model, dataset, device, batch_size=32, num_workers=0, desc='推理'):
    """按原有顺序返回整个数据集的 logits 与标签。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits, labels = [], []
    for images, targets in tqdm(loader, desc=desc, leave=False):
        logits.append(model(images.to(device)).float().cpu())
        labels.append(targets)
    if not logits:
        raise ValueError('评估数据集为空')
    return torch.cat(logits), torch.cat(labels)


def summarize(logits, labels, images_per_id=None):
    """汇总交叉熵、Top-k 和同一手掌多图平均概率的识别率。"""
    if logits.shape[0] != labels.shape[0] or logits.shape[0] == 0:
        raise ValueError('logits 与标签数量不一致或为空')

    classes = logits.shape[1]
    k = min(5, classes)
    total = labels.shape[0]
    correct = (logits.argmax(1) == labels).sum().item()
    top_correct = (logits.topk(k).indices == labels[:, None]).any(1).sum().item()
    result = {
        'loss': F.cross_entropy(logits, labels).item(),
        'correct': correct,
        'total': total,
        'top_k': k,
        'top1_pct': 100 * correct / total,
        f'top{k}_pct': 100 * top_correct / total,
    }

    if images_per_id:
        prob_sums = torch.zeros(classes, classes)
        counts = torch.zeros(classes, dtype=torch.long)
        prob_sums.index_add_(0, labels, logits.softmax(dim=1))
        counts.index_add_(0, labels, torch.ones_like(labels))
        # 只用凑满 images_per_id 张图的手掌分组计算融合识别率。
        group_labels = (counts == images_per_id).nonzero(as_tuple=True)[0]
        if len(group_labels):
            scores = prob_sums[group_labels] / images_per_id
            hits = scores.argmax(1) == group_labels
            top_hits = (scores.topk(k).indices == group_labels[:, None]).any(1)
            result['group_size'] = images_per_id
            result['group_total'] = len(group_labels)
            result['group_top1_pct'] = 100 * hits.sum().item() / len(group_labels)
            result[f'group_top{k}_pct'] = 100 * top_hits.sum().item() / len(group_labels)
    return result


def print_result(result, name=None):
    """按训练日志的措辞打印单张和融合识别率。"""
    k = result['top_k']
    prefix = f'{name}：' if name else ''
    print(f"{prefix}单张图片 Top-1 {result['top1_pct']:.2f}%"
          f"（{result['correct']}/{result['total']}），Top-{k} {result[f'top{k}_pct']:.2f}%")
    if 'group_top1_pct' in result:
        print(f"{prefix}每只手掌 {result['group_size']} 张图平均 Top-1 "
              f"{result['group_top1_pct']:.2f}%，Top-{k} {result[f'group_top{k}_pct']:.2f}%")


def evaluate(model, dataset, device, batch_size=32, images_per_id=None,
             num_workers=0, desc='评估', name=None):
    """采集整个数据集的 logits 并汇总识别率。"""
    logits, labels = collect_logits(model, dataset, device, batch_size=batch_size,
                                    num_workers=num_workers, desc=desc)
    result = summarize(logits, labels, images_per_id=images_per_id)
    print_result(result, name=name)
    return result
