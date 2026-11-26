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

`attacks/` 提供四种无目标攻击（CW 含 L∞ / L2 两个版本），统一在 `[0, 1]` 灰度像素空间工作，模型所需的 ImageNet 归一化由包装层完成：

| 名称 | 范数 | 说明 |
| --- | --- | --- |
| `pgd` | L∞ | 随机起点 + 交叉熵梯度上升，投影回 epsilon 球 |
| `bim` | L∞ | I-FGSM，从原图逐步累加梯度符号 |
| `cw` | L∞ | 默认 CW：在 epsilon 球内最大化 CW 边界损失 |
| `cw-l2` | L2 | 论文原始形式：tanh 空间 + Adam，不使用 epsilon |
| `apgd` | L∞ | Auto-PGD（APGD-CE），自适应步长与最优解跟踪 |

```powershell
python scripts/generate_adv.py --attack cw --model resnet18 --epsilon 4
python scripts/generate_adv.py --attack cw-l2 --model vgg16 --c 1 --steps 50
python scripts/generate_adv.py --attack pgd --model resnet18 --epsilon 4
python scripts/generate_adv.py --attack apgd --model resnet18 --steps 30
```

`--epsilon`、`--alpha` 的单位是 1/255（`--epsilon 4` 即 4/255），不传时读取 `configs/default.yaml` 的 `attacks` 段。结果保存在 `outputs/adv/<攻击>/<模型>/<强度>/session<批次>/`：对抗样本为 PNG（与原图同名），同目录的 `metadata.csv` 记录逐图原标签、干净/对抗预测、是否成功和 L∞ 扰动，`run_info.json` 与 `summary.csv` 记录本次参数和汇总指标。

汇总指标有两种口径，不要混用：

- `attack_success_pct`（条件 ASR）：分母是干净预测正确的图片数，分子要求干净预测正确且对抗预测错误；原本模型就分错的图片不计入分子，也不计入分母。
- `all_misclassified_pct`：全样本口径，分母是全部图片，包含原本就分错的图片。`adv_top1_pct` 同样按全部样本统计，因此不能把 `100 - adv_top1_pct` 当作攻击成功率。

已经生成过的结果不需要重新跑攻击：`python scripts/backfill_metrics.py` 会从 `metadata.csv` 重算并写回每次运行的 `summary.csv` 与 `run_info.json`；加 `--combined` 会再生成 `outputs/adv/all_runs.csv` 横向汇总表，`--dry-run` 只检查不写文件。

## 低通嵌入相似度

分析原图与高斯低通滤波图像的嵌入余弦相似度随衰减带宽半径 `r` 的变化。默认每个模型
单独一张图，图中包含干净样本和 `outputs/adv` 里每种攻击的全部预算（PGD/BIM/APGD 的
ε=1、ε=4，CW 的当前运行）；同一种攻击共用颜色和标记，预算用线型区分（ε 越大越接近实线），
横轴半径由大到小：

```powershell
python scripts/analyze_lpf_similarity.py                  # 两个模型，session2 全部 6000 张
python scripts/analyze_lpf_similarity.py --limit 1500 --balance-classes  # 抽样并覆盖全部 600 类
python scripts/analyze_lpf_similarity.py --epsilon 4      # 只保留 ε=4 预算
python scripts/analyze_lpf_similarity.py --attack pgd --attack cw
python scripts/analyze_lpf_similarity.py --no-attacks     # 只看干净样本
```

低通滤波在 `[0, 1]` 像素空间进行，滤波后再做 ImageNet 归一化；嵌入取分类层之前的表示
（ResNet-18 为 512 维，VGG-16 为 4096 维）。结果写入
`results/lpf_similarity/session<批次>_<滤波器>_<时间>_<模型>.png/.pdf` 和同名的汇总 CSV
（含每个条件、每个半径的均值、标准差、5%/50%/95% 分位）。图中只给干净样本画 ±1 标准差
阴影，避免多条曲线的阴影互相遮挡；`--no-save` 只打印表格。

## 深度特征几何分析

分析干净掌纹、PGD/BIM/CW/APGD 对抗样本与普通随机噪声样本（默认 L∞ 球内均匀噪声）的深度特征：
用 session1（训练批次）的同身份样本建立每个身份的“真实身份中心”和局部 PCA 子空间，
再统计 session2 各条件的类内距离、类间距离、到真实身份中心的距离、到最近错误身份
中心的距离，以及到本身份局部子空间的重构残差；同时用 PCA、t-SNE 可视化：

```powershell
python scripts/analyze_feature_geometry.py                 # 两个模型，session2 全部 6000 张
python scripts/analyze_feature_geometry.py --limit 1200 --balance-classes
python scripts/analyze_feature_geometry.py --model resnet18 --epsilon 4
python scripts/analyze_feature_geometry.py --attack pgd --attack cw --noise-level 1 4
python scripts/analyze_feature_geometry.py --noise-mode gaussian --noise-level 1
python scripts/analyze_feature_geometry.py --subspace-variance 0.99 --per-sample
python scripts/analyze_feature_geometry.py --no-save       # 只打印表格
```

对抗样本读取 `outputs/adv/<攻击>/<模型>/<预算>/session<批次>/`，随机噪声在读取时在线
生成并量化到 8 位。默认 `--noise-mode uniform`：每个像素在 `[-ε, ε]` 内均匀采样，
`--noise-level`（单位 1/255，默认 4）就是 L∞ 预算，因此与 PGD/BIM/APGD 的逐像素
最大扰动一致，是公平的普通噪声对照。也可以设 `--noise-mode gaussian`，此时
`--noise-level` 是标准差 σ；注意同样 4/255 下高斯噪声的 L2 范数约 6.1，而
PGD ε=4/255 约 4.1、BIM 约 2.5、CW 约 1.5，高斯噪声在 L2 上是更强的扰动，
会显著拉低识别率，跨条件比较时需要留意。
距离同时输出余弦距离和欧氏距离；类内距离是同一条件下每个身份内部两两距离的均值再对
身份等权平均，类间距离是该条件下身份中心之间的平均距离。“真实身份中心”只用 session1
训练样本计算，因此干净样本也会有非零距离，常用于对比对抗/噪声特征偏离了多少。
局部 PCA 子空间的维数由 `--subspace-variance`（默认 0.95）控制，最多到同身份训练
样本数减一；重构残差同时给出原始欧氏残差和 `残差/到中心距离` 的比值，后者约等于
偏离方向落在子空间外的比例。需要注意，每个身份只有 10 张训练图、子空间最多 9 维，
所以干净样本的比值本身也会偏高（实测约 0.88）；这个指标应与干净样本或其他条件
横向比较，而不是看绝对值。

每个模型输出两张图：PCA 与 t-SNE 全局分布图（颜色区分样本类型，空心表示较弱预算或
噪声强度），以及若干身份的局部 PCA 放大图（白色星号为该身份的训练中心）。可视化默认
抽取 60 个身份、每个条件每类最多 4 张，可用 `--viz-classes`、`--viz-per-class`、
`--zoom-classes` 调整。结果写入 `results/feature_geometry/`：

- `distances_session<批次>_<时间>.csv`：一行一个「模型 × 条件」的距离指标；
- `residuals_session<批次>_<时间>.csv`：局部子空间重构残差与到中心距离；
- `--per-sample` 另存逐图距离与残差明细；
- `session<批次>_<时间>_<模型>_pca_tsne.png/.pdf` 和 `_class_zoom.png/.pdf`；
- `features/<时间>_session<批次>_<模型>/`：保存的深度特征，见下。

每个模型的特征目录包含逐条件的 `clean.npz`、`pgd-eps1.npz`、`cw-c1_kappa0.npz`、
`noise-uniform-4.npz` 等，每个 npz 有 `features [N, D]`、`labels [N]`、
`filenames [N]`；另有 `train_session1.npz`（训练参考特征）、`reference.npz`
（身份中心 `centers [600, D]`、局部 PCA 基及其 `basis_offsets`）和
`manifest.json`（模型、预处理、条件与文件对应关系、运行参数）。保存的是分类层输入
的原始嵌入，未做 L2 归一化，与脚本内算距离/PCA 用的数值完全一致。默认 float32，
VGG-16 每个条件约 100 MB；可用 `--feature-dtype float16` 减半，或
`--no-save-features` 关闭。直接复用示例：

```python
import json
from pathlib import Path

from utils.features import load_embeddings, load_identity_reference

run = Path('results/feature_geometry/features/20261125_193000_session2_resnet18')
manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))

clean = load_embeddings(run / 'clean.npz')
features, labels, filenames = clean['features'], clean['labels'], clean['filenames']

reference = load_identity_reference(run / 'reference.npz')
centers = reference['centers']          # [600, D]
basis = reference['bases'][c]           # 身份 c 的局部 PCA 基 [r, D]
```

提取与保存也封装在 [utils/features.py](utils/features.py)，可直接在其它分析里复用：

```python
from utils.features import extract_embeddings, save_embeddings

features, labels = extract_embeddings(model, dataset, 'cuda', batch_size=64)
save_embeddings('results/features/clean.npz', features, labels,
                [path.name for path, _ in dataset.samples])
```

## 位平面噪声分析（LOTA）

借鉴 `learn_from/LOTA` 的位平面分解：取每张图最低 k 位并拉伸成噪声图，再按 LOTA
的局部差分能量（水平、垂直、两条对角线相邻像素绝对差之和）切块打分。
`defense/bitplane.py` 把原实现改写为 `[N, C, H, W]`、取值 `[0, 1]` 的批量张量接口；
`scripts/analyze_bitplane_noise.py` 用它对比干净样本与 `outputs/adv` 里的对抗样本。
只做图像统计、不做模型前向，因此不需要 GPU：

```powershell
python scripts/analyze_bitplane_noise.py                    # 两个模型，session2 全部数据
python scripts/analyze_bitplane_noise.py --limit 1200 --balance-classes
python scripts/analyze_bitplane_noise.py --bits 1 --patch-size 16
python scripts/analyze_bitplane_noise.py --epsilon 4 --attack pgd
python scripts/analyze_bitplane_noise.py --patch-mode random --num-patches 49
python scripts/analyze_bitplane_noise.py --export-dir outputs/bitplane --export-limit 16
```

`--bits` 默认 3、`--patch-mode max` 与 LOTA 一致；加 `--num-patches` 会像 LOTA 那样
随机裁块（`--patch-mode random` 即 LOTA 训练时的设置），默认是非重叠切块，结果可复现。
每个模型生成一张图：左图是逐图最强噪声块能量的箱线图，右图是各位平面置 1 比例
相对干净样本的变化，可以看出对抗扰动落在哪一位（例如 `ε=1/255` 的 PGD 主要改变 LSB）。
汇总 CSV 写入 `results/bitplane_noise/session<批次>_bits<k>_<时间>.csv`，一行一个
「条件 × 指标」；`--per-image` 另存逐图明细，`--export-dir` 把选中的噪声块放大后
存成 PNG，便于直接查看或作为后续分类器的输入。

在代码中也可直接调用：

```python
from defense.bitplane import bit_plane_energy, bitplane_noise, low_bit_plane

noise = low_bit_plane(images, bits=3, mode='scaling')    # 低位平面噪声图
patch = bitplane_noise(images, bits=3, patch_size=32, patch_mode='max')
energy = bit_plane_energy(images, bits=8)                # [N, 8]，第 0 列是 LSB
```

## 多尺度多方向 Gabor 方向响应分析

`defense/gabor.py` 用一组不同波长 λ、不同载波方向的解析 Gabor 滤波器提取掌纹
局部方向响应，`scripts/analyze_gabor_robustness.py` 对比干净样本与 `outputs/adv`
中的对抗样本。只做图像统计、不做模型前向，因此不需要加载模型权重；有 GPU 时
FFT 会快很多。动机来自掌纹识别中的 CompCode：方向码本身具有身份判别力，因此
可以问“哪些尺度和方向的方向码在对抗扰动下仍然稳定”，把稳定的方向码当作身份锚点。

实现要点：

- 方向：θ 指 Gabor 载波方向（周期 180°）；纹线方向 = θ + 90°。竞争编码取
  `--dominance min` 时选响应最小的滤波器，其载波方向等于纹线方向，即 CompCode
  式的最小响应规则（本文用解析 Gabor 幅度实现）；默认 `max` 取响应最强的滤波器，
  其载波方向是纹线法向，实测在细尺度上更稳定。两种规则是同一滤波器组上的不同
  竞争判决，方向语义相差 90°，可在同一数据上直接对比稳定性，但解释具体角度时
  要区分。
- 尺度：默认 λ=4/6/8/12/16/24，σ=0.56λ，约一个倍频程带宽；表格中的
  `structure` 按 λ<8 细粒度纹理、8≤λ<16 褶皱、λ≥16 掌纹主线粗略分组。
- 频域实现：Gabor 核的频域响应是中心在 `(cosθ/λ, sinθ/λ)` 的高斯，直接在 FFT 域
  乘掩膜再逆变换；图像先反射填充（默认 3 倍最大包络标准差）以减少循环边界，
  并去掉直流分量。滤波器组一次构造、按批复用。

```powershell
python scripts/analyze_gabor_robustness.py                    # 两个模型，全部 6000 张
python scripts/analyze_gabor_robustness.py --limit 1200 --balance-classes
python scripts/analyze_gabor_robustness.py --dominance min    # CompCode 式最小响应
python scripts/analyze_gabor_robustness.py --wavelengths 4 8 16 24 --orientations 8
python scripts/analyze_gabor_robustness.py --model resnet18 --epsilon 4
python scripts/analyze_gabor_robustness.py --no-identity --no-save
```

`--angles` 可以直接给出方向列表（覆盖 `--orientations`）。`--epsilon` 只保留某个
L∞ 预算；CW 目录下同时存在 eps1、eps4 和 c1_kappa0（实际是 CW-L2），脚本会把
三个条件都纳入，CW-L2 在表格中单独标注。

在本项目的 session1→session2 检索中，`max` 的联合和单尺度 Rank-1 都高于
`min`；`min` 的同图锚点余弦略高，但单尺度身份区分度明显更低。因此 `max` 更
适合作为默认的身份锚点，`min` 主要用于和 CompCode 规则对照。

指标口径：

- `magnitude_corr`：同一图、同一滤波器下干净/对抗响应图的逐像素 Pearson 相关。
- `energy_ratio_db`：对抗/干净响应能量的 dB 比；高频对抗扰动通常让细尺度能量升高。
- `code_agreement`：同一图、同一尺度下方向码相同的像素比例。
- `code_retention`：P(对抗码 = k | 干净码 = k)，按方向统计；`mean/std` 为逐图等权，
  `pooled` 为按像素数加权。`transitions.csv` 给出完整的 from→to 条件概率，可看
  方向码是局部翻转还是近似随机跳变。
- `identity`：把方向码按 G×G 分块统计直方图作为身份描述子，查询为当前条件、
  图库为另一 session 的干净图，报告 Rank-1/Rank-5；`anchor_cosine` 是同一图
  干净/对抗描述子的余弦。特征集合包括全部尺度×方向、单尺度、单方向、单尺度单方向。

结果写入 `results/gabor_robustness/`：

- `session2_<规则>_<时间>_robustness.csv`：一行一个「条件 × 尺度 × 方向」；
- `session2_<规则>_<时间>_scale.csv`：一行一个「条件 × 尺度」（码一致率、角误差等）；
- `session2_<规则>_<时间>_identity.csv`：一行一个「条件 × 特征集合」；
- `session2_<规则>_<时间>_transitions.csv`：方向码转移概率；
- `session2_<规则>_<时间>_images.csv`：逐图指标（仅 `--per-image`）；
- `session2_<规则>_<时间>_run_info.json`：本次参数与滤波器组配置；
- `<模型>_retention.png` / `_corr.png`：尺度×方向的码保留率、响应相关性热图；
- `<模型>_curves.png`：码一致率/码保留率/响应相关性随尺度、方向的变化；
- `<模型>_identity.png`：Rank-1、锚点余弦随尺度变化，以及干净/最差条件的
  单尺度单方向 Rank-1 和下降热图；
- `<模型>_transition.png`：各条件的方向码转移矩阵。

在代码中可直接调用：

```python
from defense.gabor import (make_gabor_bank, gabor_magnitude,
                           dominant_orientation, orientation_histogram)

bank = make_gabor_bank(224, 224, wavelengths=[4, 8, 16, 24], orientations=6)
magnitude = gabor_magnitude(images, bank)                  # [N, K, H, W]
codes = dominant_orientation(magnitude, bank, rule='max')  # [N, S, H, W]
hist = orientation_histogram(codes, bank, grid=4)          # [N, S, G, G, O]
```

`gabor_magnitude` 只返回幅度；需要相位时用 `gabor_response` 取复响应（实部/虚部
对应偶/奇对称滤波器）。`make_gabor_bank` 默认把掩膜放在 CPU，传入
`device='cuda'` 可整组放到 GPU。

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
