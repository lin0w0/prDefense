"""按名称登记并加载 models/train_weights 下已训练的掌纹模型。"""

from pathlib import Path

import torch
import yaml

from models.loader import load_model

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / 'configs' / 'default.yaml'


def select_device(preference=None):
    """None 或 'auto' 自动选择 GPU；也可指定 'cpu'、'cuda' 或 'cuda:1'。"""
    if preference in (None, 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(preference)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定了 CUDA 设备，但 PyTorch 未检测到可用 GPU')
    return device


def resolve_path(value, base=PROJECT):
    """配置文件中的相对路径以项目根目录为准。"""
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path=DEFAULT_CONFIG):
    """读取模型登记与评估设置，并检查必要字段。"""
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as file:
        config = yaml.safe_load(file)
    models = config.get('models') if isinstance(config, dict) else None
    if not isinstance(models, dict) or not models:
        raise ValueError(f'{path} 需要非空的 models 段')
    for key, entry in models.items():
        if not isinstance(entry, dict) or not entry.get('weights') or not entry.get('arch'):
            raise ValueError(f'models.{key} 需要 arch 和 weights 字段')
    if not isinstance(config.get('evaluation'), dict):
        config['evaluation'] = {}
    return config


def available_models(config_path=DEFAULT_CONFIG):
    """返回 {名称: 登记信息}，便于脚本列出全部模型。"""
    return load_config(config_path)['models']


def load_pretrained(name, device=None, config_path=DEFAULT_CONFIG, config=None):
    """按名称加载默认配置中的模型；也可直接传入权重路径，结构从权重推断。"""
    config = config or load_config(config_path)
    entry = config['models'].get(name)
    if entry is None:
        path = resolve_path(name)
        entry = {'arch': None, 'weights': str(path), 'description': None}
    else:
        path = resolve_path(entry['weights'])

    model, info = load_model(path, device=select_device(device), arch=entry.get('arch'))
    info = dict(info, name=name, weights=str(path), description=entry.get('description'))
    return model, info
