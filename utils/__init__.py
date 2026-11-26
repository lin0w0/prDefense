"""通用工具模块。"""

from .features import (
    extract_embeddings, load_embeddings, load_identity_reference,
    normalize_images, save_embeddings, save_identity_reference,
)

__all__ = [
    'extract_embeddings',
    'load_embeddings',
    'load_identity_reference',
    'normalize_images',
    'save_embeddings',
    'save_identity_reference',
]
