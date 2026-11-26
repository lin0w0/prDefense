"""深度嵌入的提取与 npz 持久化，供各个分析脚本复用。

嵌入取分类层输入（ResNet-18 为 512 维，VGG-16 为 4096 维），输入先做
ImageNet 归一化；单通道灰度图会复制成三通道。保存时直接写原始嵌入，
不做 L2 归一化或标准化，读回来后可直接用于距离、PCA 或局部子空间计算。

用法：
    from utils.features import extract_embeddings, save_embeddings, load_embeddings

    features, labels = extract_embeddings(model, dataset, 'cuda')
    save_embeddings('features/clean.npz', features, labels, filenames)
    data = load_embeddings('features/clean.npz')   # features / labels / filenames
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import IMAGENET_MEAN, IMAGENET_STD
from defense.features import EmbeddingExtractor

FEATURE_DTYPES = ('float32', 'float16')


def normalize_images(images, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """把 [N, C, H, W]、取值 [0, 1] 的像素图归一化为模型输入。

    单通道输入会复制成三通道，与训练时的 Grayscale(num_output_channels=3) 一致。
    """
    if isinstance(images, torch.Tensor) and images.dim() != 4:
        raise ValueError(f'期望 [N, C, H, W] 输入，实际为 {tuple(images.shape)}')
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    device = images.device
    mean_tensor = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return (images - mean_tensor) / std_tensor


@torch.no_grad()
def extract_embeddings(model, dataset, device, batch_size=64, num_workers=0,
                       desc='提取特征'):
    """按数据集顺序提取分类层输入嵌入，返回 (features [N, D], labels [N])。"""
    if len(dataset) == 0:
        raise ValueError('数据集为空')
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    features, labels = [], []
    with EmbeddingExtractor(model) as extractor:
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            features.append(extractor(normalize_images(images)).float().cpu())
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)


def _as_numpy(value):
    """把 torch 张量或任意数组转换成 numpy，不改设备上的原值。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_embeddings(path, features, labels, filenames=None, dtype='float32'):
    """保存一个条件的嵌入、标签和可选文件名到 npz。

    path 不带 .npz 时会自动补上；dtype 可选 float32（默认）或 float16。
    返回实际写入的路径。
    """
    if dtype not in FEATURE_DTYPES:
        raise ValueError(f'不支持的特征精度 {dtype!r}，可选 {", ".join(FEATURE_DTYPES)}')
    path = Path(path)
    if path.suffix.lower() != '.npz':
        path = path.with_suffix('.npz')
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        'features': _as_numpy(features).astype(dtype, copy=False),
        'labels': _as_numpy(labels).astype(np.int64, copy=False),
    }
    if filenames is not None:
        arrays['filenames'] = np.asarray(filenames)
    np.savez(path, **arrays)
    return path


def load_embeddings(path):
    """读取 save_embeddings 写出的 npz，返回 features/labels/filenames。"""
    with np.load(path) as data:
        result = {
            'features': data['features'],
            'labels': data['labels'],
        }
        if 'filenames' in data:
            result['filenames'] = data['filenames']
    return result


def save_identity_reference(path, centers, bases, present, variance, mean_rank):
    """保存身份中心与局部 PCA 基。

    centers [C, D]；bases 是每个身份的 [r, D] 基组成、长度 C 的列表；
    present [C] 标记训练集中是否出现该身份。所有基按身份顺序拼成
    basis_values，并用 basis_offsets 记录切分位置，读回时自动还原成列表。
    """
    path = Path(path)
    if path.suffix.lower() != '.npz':
        path = path.with_suffix('.npz')
    path.parent.mkdir(parents=True, exist_ok=True)

    centers = _as_numpy(centers).astype(np.float32, copy=False)
    offsets = np.zeros(len(bases) + 1, dtype=np.int64)
    pieces = []
    for index, basis in enumerate(bases):
        array = _as_numpy(basis).astype(np.float32, copy=False)
        pieces.append(array)
        offsets[index + 1] = offsets[index] + array.shape[0]
    if pieces:
        basis_values = np.concatenate(pieces, axis=0)
    else:
        basis_values = np.zeros((0, centers.shape[1]), dtype=np.float32)

    np.savez(
        path,
        centers=centers,
        present=_as_numpy(present),
        ranks=np.array([_as_numpy(basis).shape[0] for basis in bases],
                       dtype=np.int64),
        basis_offsets=offsets,
        basis_values=basis_values,
        subspace_variance=np.float32(variance),
        mean_rank=np.float32(mean_rank),
    )
    return path


def load_identity_reference(path):
    """读取身份中心与局部 PCA 基，bases 还原为逐身份的 [r, D] 数组列表。"""
    with np.load(path) as data:
        offsets = data['basis_offsets']
        values = data['basis_values']
        bases = [values[offsets[index]:offsets[index + 1]]
                 for index in range(len(offsets) - 1)]
        return {
            'centers': data['centers'],
            'present': data['present'],
            'ranks': data['ranks'],
            'bases': bases,
            'subspace_variance': float(data['subspace_variance']),
            'mean_rank': float(data['mean_rank']),
        }
