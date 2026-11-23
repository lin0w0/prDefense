# prDefense

研究基于掌纹识别分类任务训练的 ResNet-18 和 VGG-16 模型在推理测试时的防御方法研究。

## 已训练模型

两个模型的权重放在 `models/train_weights`，并在 `configs/default.yaml` 中登记：

| 名称 | 网络 | 权重 | 训练记录 |
| --- | --- | --- | --- |
| `resnet18` | ResNet-18 | `models/train_weights/ResNet-18/best_model.pth` | 98.62% |
| `vgg16` | VGG-16 | `models/train_weights/VGG-16/best_model.pth` | 86.03% |

均为同济掌纹 600 类：session1 训练、session2 验证。`models/loader.py` 按权重中的 `config.model` 或参数名重建网络，`models/manager.py` 负责按名称解析路径和选择设备。

## 环境

```powershell
D:/code/envs/prDefense/python.exe -m pip install -r requirements.txt
```

## 评估

数据放在 `data/resources/Tongji/ROI/session{1,2}`，默认评估 session2（与训练时的验证批次一致）：

```powershell
python scripts/evaluate.py                    # 评估两个模型
python scripts/evaluate.py --model resnet18   # 只评估一个
python scripts/evaluate.py --session 1        # 评估训练批次
python scripts/evaluate.py --limit 100        # 只取前 100 张图快速自检
python scripts/evaluate.py --num-workers 4    # 加快图片读取
```

脚本打印单张 Top-1/Top-5 和同一只手掌 10 张图平均概率的融合识别率，并用 pandas 把一行一个模型的结果表写入 `results/evaluate/session<批次>-<时间>.csv`（UTF-8 BOM，Excel 可直接打开）。

## 在代码中加载

```python
from data.dataset import make_session_dataset
from evaluations.metrics import evaluate, collect_logits
from models.manager import load_pretrained

model, info = load_pretrained('resnet18')          # 也可以传权重路径
dataset = make_session_dataset(2)                  # 标签为 (编号-1)//10
result = evaluate(model, dataset, 'cuda', images_per_id=10)

logits, labels = collect_logits(model, dataset, 'cuda')  # 对抗评估可直接复用
```

对抗攻击与防御代码建议只依赖 `load_pretrained`、`make_session_dataset` 和 `collect_logits`/`summarize`，保持与训练时相同的 224×224、ImageNet 归一化和标签顺序。
