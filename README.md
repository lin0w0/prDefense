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

## 对抗样本生成

`attacks/` 提供四种无目标攻击，统一在 `[0, 1]` 灰度像素空间工作，模型所需的 ImageNet 归一化由包装层完成：

| 名称 | 范数 | 说明 |
| --- | --- | --- |
| `pgd` | L∞ | 随机起点 + 交叉熵梯度上升，投影回 epsilon 球 |
| `bim` | L∞ | I-FGSM，从原图逐步累加梯度符号 |
| `cw` | L2 | tanh 空间 + Adam，优化 CW 边界损失 |
| `apgd` | L∞ | Auto-PGD（APGD-CE），自适应步长与最优解跟踪 |

```powershell
python scripts/generate_adv.py --attack pgd --model resnet18 --epsilon 4
python scripts/generate_adv.py --attack bim --model resnet18 --limit 100
python scripts/generate_adv.py --attack cw --model vgg16 --c 1 --steps 50
python scripts/generate_adv.py --attack apgd --model resnet18 --steps 30
```

`--epsilon`、`--alpha` 的单位是 1/255（`--epsilon 4` 即 4/255），不传时读取 `configs/default.yaml` 的 `attacks` 段。结果保存在 `outputs/adv/<攻击>/<模型>/<强度>/session<批次>/`：对抗样本为 PNG（与原图同名），同目录的 `metadata.csv` 记录逐图原标签、干净/对抗预测、是否成功和 L∞ 扰动，`run_info.json` 与 `summary.csv` 记录本次参数和汇总指标。

汇总指标有两种口径，不要混用：

- `attack_success_pct`（条件 ASR）：分母是干净预测正确的图片数，分子要求干净预测正确且对抗预测错误；原本模型就分错的图片不计入分子，也不计入分母。
- `all_misclassified_pct`：全样本口径，分母是全部图片，包含原本就分错的图片。`adv_top1_pct` 同样按全部样本统计，因此不能把 `100 - adv_top1_pct` 当作攻击成功率。

已经生成过的结果不需要重新跑攻击：`python scripts/backfill_metrics.py` 会从 `metadata.csv` 重算并写回每次运行的 `summary.csv` 与 `run_info.json`；加 `--combined` 会再生成 `outputs/adv/all_runs.csv` 横向汇总表，`--dry-run` 只检查不写文件。

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
