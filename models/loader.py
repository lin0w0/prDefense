"""构建网络结构并从 prResNet-18 / prVGG-16 保存的权重恢复模型。"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18, vgg16

NUM_CLASSES = 600


def build_resnet18(num_classes=NUM_CLASSES):
    """三通道掌纹分类模型，结构需与 prResNet-18/model.py 一致。"""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_vgg16(num_classes=NUM_CLASSES):
    """三通道掌纹分类模型，结构需与 prVGG-16/model.py 一致。"""
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


BUILDERS = {'resnet18': build_resnet18, 'vgg16': build_vgg16}


def infer_architecture(state_dict):
    """按参数名判断网络结构，兼容不含 config 的纯 state_dict 权重。"""
    keys = state_dict.keys()
    if 'conv1.weight' in keys and 'layer4.1.conv2.weight' in keys:
        return 'resnet18'
    if 'features.0.weight' in keys and 'classifier.6.weight' in keys:
        return 'vgg16'
    raise ValueError('无法从参数名判断网络结构，请通过 arch 参数指定 resnet18 或 vgg16')


def infer_num_classes(state_dict, architecture):
    """分类层输出数即手掌类别数，直接从权重形状读取。"""
    key = 'fc.weight' if architecture == 'resnet18' else 'classifier.6.weight'
    return int(state_dict[key].shape[0])


def read_checkpoint(path, arch=None):
    """读取 best_model.pth，返回 state_dict 和权重中记录的元信息。

    训练脚本保存的是 {'model': state_dict, 'num_classes': ..., 'config': ...,
    'epoch': ..., 'val_loss': ..., 'val_acc': ...}；也接受只含 state_dict 的文件。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'找不到权重文件：{path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get('model'), dict):
        state_dict = checkpoint['model']
        metadata = checkpoint
    elif isinstance(checkpoint, dict) and checkpoint and all(
            torch.is_tensor(value) for value in checkpoint.values()):
        state_dict = checkpoint
        metadata = {}
    else:
        raise ValueError(f'{path} 不是可识别的模型权重文件')

    config = metadata.get('config') or {}
    architecture = arch or config.get('model') or infer_architecture(state_dict)
    if architecture not in BUILDERS:
        raise ValueError(f'不支持的网络结构 {architecture!r}，可选 {", ".join(BUILDERS)}')

    info = {
        'architecture': architecture,
        'num_classes': infer_num_classes(state_dict, architecture),
        'epoch': metadata.get('epoch'),
        'val_loss': metadata.get('val_loss'),
        'val_acc': metadata.get('val_acc'),
        'config': config,
    }
    return state_dict, info


def load_model(path, device='cpu', arch=None):
    """恢复一个处于评估模式的模型，并返回权重中记录的元信息。"""
    state_dict, info = read_checkpoint(path, arch=arch)
    model = BUILDERS[info['architecture']](info['num_classes'])
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, info
