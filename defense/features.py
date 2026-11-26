"""提取分类模型倒数第二层的嵌入特征。

ResNet-18 取 fc 的输入（512 维），VGG-16 取 classifier 最后一层的输入
（4096 维）。这些都是分类头之前的整图表示，适合做余弦相似度比较。
"""

import torch
import torch.nn as nn


def final_classifier(model):
    """返回分类模型的最后一层线性层。"""
    fc = getattr(model, 'fc', None)
    if isinstance(fc, nn.Linear):
        return fc
    classifier = getattr(model, 'classifier', None)
    if isinstance(classifier, nn.Sequential) and isinstance(classifier[-1], nn.Linear):
        return classifier[-1]
    raise ValueError('无法定位分类层，期望模型包含 Linear 类型的 fc 或 classifier[-1]')


class EmbeddingExtractor:
    """前向一次模型并返回分类层输入，即嵌入特征 [N, D]。

    用法：
        extractor = EmbeddingExtractor(model)
        features = extractor(images)   # images 已完成归一化
        extractor.close()
    也可以作为上下文管理器使用。
    """

    def __init__(self, model):
        self.model = model
        self.classifier = final_classifier(model)
        self.embedding_dim = self.classifier.in_features
        self._captured = None
        self._handle = self.classifier.register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        self._captured = inputs[0]

    @torch.no_grad()
    def __call__(self, images):
        self._captured = None
        self.model(images)
        if self._captured is None:
            raise RuntimeError('没有捕获到嵌入特征，请检查模型前向路径')
        return self._captured

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
