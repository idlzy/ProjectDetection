# ProjectDetection

独立 PyTorch 实现的 MW3D 单目 3D 检测工程，覆盖数据检查、FCOS3D/PGDA
建模、训练、验证、测试、推理和 ONNX 导出。运行时不依赖 HAT 或
`horizon_plugin_pytorch`。

当前实现是可运行的第一阶段基线。网络、数据流和命令行已经形成闭环，并提供
旧 HAT 1.8.2 EfficientNet-B0 + BiFPN + PGDA float checkpoint 的测试兼容模型。

## 环境

远程训练环境基线为 Python 3.8、PyTorch 1.13 和 CUDA 11.6：

```bash
python3 -m pip install -e '.[dev]'
```

数据不复制进仓库。容器内默认路径：

```text
/data/horizon_j5/data/mw3d_v5_data
/data/horizon_j5/data/mw3d_ready_v5
```

## 使用

本地 `/home/gy-zb-a-luziyang/datasets/mw3d` 数据首次使用时，先按完整时序组生成
训练、验证和测试清单。默认比例为 8:1:1，固定随机种子为 3407：

```bash
python3 tools/split_dataset.py --dry-run
python3 tools/split_dataset.py

python3 tools/analyze_dataset.py \
  --config configs/experiments/fcos3d_exp_pgda_local.yaml
```

脚本读取根目录的 `frames.jsonl`，输出 `splits/train_frames.jsonl`、
`val_frames.jsonl` 和 `test_frames.jsonl`。它先按 `image_rel` 中的
`images/<大类目录>/...` 分层，再在每个大类内部以完整 `split_group` 为单位尽量
按 8:1:1 划分。因此每类数据都会尽量覆盖三个集合，同时不会将同一段连续采集
数据拆进不同集合。每个大类的实际帧数和组数记录在 `split_summary.json`；若输出
已经存在，只有显式添加 `--force` 才会覆盖。

标定读取规则与 HAT 保持一致：split manifest 的 `extrinsic_rel` 非空时，
优先使用该逐帧 vehicle-to-camera 外参；没有 `extrinsic_rel` 时才使用
`FrontViewCalibParam.json` 内嵌外参。若 manifest 声明了外参文件但文件缺失或
格式错误，加载会直接报错，不会静默回退。训练日志会统计两种标定来源的样本数。

`analyze_dataset.py` 一次扫描完成 split 泄漏检查、帧与标注框计数、大小和距离
分布、类别/小类/ann_batch 统计以及 visibility 质量统计，并输出 JSON、CSV 和中文
图表。默认检测到 train/val/test 泄漏时返回非零状态。

先做数据检查和小样本冒烟训练：

```bash
python tools/analyze_dataset.py --config configs/experiments/fcos3d_exp_pgda.yaml

CUDA_VISIBLE_DEVICES=0,1 scripts/start_train.sh \
  --config configs/experiments/fcos3d_exp_pgda.yaml \
  --set data.batch_size_per_gpu=4 runtime.max_train_samples=32 \
        runtime.max_val_samples=16 train.epochs=1
```

完整训练、验证和测试：

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/start_train.sh \
  --config configs/experiments/fcos3d_exp_pgda.yaml

python tools/validate.py --config configs/experiments/fcos3d_exp_pgda.yaml \
  --checkpoint outputs/fcos3d_mw3d_ready_v5_bgr_exp_pgda/checkpoints/best.pth

python tools/test.py --config configs/experiments/fcos3d_exp_pgda.yaml \
  --checkpoint outputs/fcos3d_mw3d_ready_v5_bgr_exp_pgda/checkpoints/best.pth
```

一次性生成 test 集可视化、正式指标 JSON 和指标图表：

```bash
CHECKPOINT=checkpoints/trained/0909/best.pth \
CONFIG=configs/experiments/fcos3d_r101_fpn_dcn_full_pgd_local.yaml \
DEVICE_ID=0 VIZ_MAX_IMAGES=20 \
scripts/run_test_viz_and_eval.sh
```

旧框架的 `float-checkpoint-last-71535492.pth.tar` 使用专用兼容配置测试：

```bash
CHECKPOINT=checkpoints/trained/0903_gy/float-checkpoint-last-71535492.pth.tar \
CONFIG=configs/experiments/fcos3d_legacy_hat_0903_local.yaml \
DEVICE_ID=0 \
VIZ_SCORE_THR=0.05 \
EVAL_SCORE_THR=0.05 \
SAVE_BEV=1 \
scripts/run_test_viz_and_eval.sh
```

将远程 `mw3d_ready_v5` 清单同步到本机后，可直接在本机使用：

```bash
CHECKPOINT=checkpoints/trained/0903_gy/float-checkpoint-last-71535492.pth.tar \
CONFIG=configs/experiments/fcos3d_legacy_hat_0903_ready_v5_local.yaml \
DEVICE_ID=0 \
VIZ_SCORE_THR=0.05 \
EVAL_SCORE_THR=0.05 \
SAVE_BEV=1 \
scripts/run_test_viz_and_eval.sh
```

该配置从本机 `/home/gy-zb-a-luziyang/datasets/mw3d_ready_v5/splits/test_frames.jsonl`
取历史划分，再到 `/home/gy-zb-a-luziyang/datasets/mw3d` 读取图片、标注和标定实体
文件。它不会修改或使用新框架在 `datasets/mw3d/splits/` 下生成的划分。测试报告
会保存实际数据路径以及 checkpoint、test manifest 的 SHA256，便于追溯与复现实验。

兼容模型保留旧 checkpoint 的 1365 个参数键及其形状，并复现旧 head 的反向
中心偏移、方向分箱解码以及“深度置信度只参与排序”的 PGDA 后处理约定。它只依赖
标准 PyTorch，不依赖 `hat` 或 `horizon_plugin_pytorch`。
在远程宿主机测试时额外设置
`DATA_ROOT=/media/datasets/lzy/mw3d`；`READY_ROOT` 默认跟随 `DATA_ROOT`，也可
单独覆盖。

脚本会自动选择包含 PyTorch、OpenCV 和 PyYAML 的 Python；也可以通过
`TEST_PYTHON=/path/to/python` 显式指定环境。

结果默认写入 `outputs/test_runs/mw3d_test_<时间>/`。可视化目录与参考脚本一致，
按 test manifest 的 `split_group` 分组，图片命名为
`visualizations/<split_group>/<原图stem>_pred.jpg`；此外还包括 `metrics.json`
以及 `plots/` 下的汇总柱状图、分类 AP 图和距离阈值 AP 矩阵。展示阈值默认
为 `0.35`，完整评估阈值默认采用 `0.05`；可分别通过 `VIZ_SCORE_THR` 和
`EVAL_SCORE_THR` 调整。旋转 NMS 固定使用项目内置的精确 CUDA 实现，不再提供
后端选择或 CPU 回退。设置 `EVAL_MAX_SAMPLES` 可进行小规模冒烟测试。
可视化默认覆盖完整 test 清单，设置 `VIZ_MAX_IMAGES` 可限量；已有图片默认跳过，
`VIZ_START_INDEX` 可从指定 manifest 下标续跑，`VIZ_OVERWRITE=1` 可覆盖，
`SAVE_BEV=1` 可额外输出 BEV 图。

测试报告的 `plots/` 目录同时生成：汇总指标、三项 TP error、mADE 诊断、各类 AP、
AP 距离阈值矩阵、带 background 的 2m 混淆矩阵、各类 101 点 PR 曲线、各类
AP/Recall/Precision 对比，以及“各类平均 AP + 深度 bin Recall + 0.5/1/2/4m
阈值 Recall”三联图，并额外提供各类别在 0.5/1/2/4m 下的分组 AP 柱状图。
每张 PNG 都有对应 JSON 数据文件。

默认统计 `data.classes` 中的全部类别。若只希望计算并绘制指定类别，可在实验配置中
设置；mAP、NDS、GT/预测/TP 数量、混淆矩阵和所有图表都会仅基于该子集：

```yaml
evaluation:
  classes: [car, truck, Pedestrian]
```

临时运行时也可以使用环境变量，不修改 YAML：

```bash
EVAL_CLASSES='[car, truck, Pedestrian]' scripts/run_test_viz_and_eval.sh
```

设为 `null` 表示恢复为全部类别。类别名称必须存在于 `data.classes`，顺序决定报告和
绘图中的显示顺序。

后台训练使用内置启停脚本，不需要把终端输出重定向到日志文件。每个实验会独立
登记占用的 GPU；实验名重复、输出目录重复或 GPU 有交集时拒绝启动，不同实验且
GPU 不同则可同时运行：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/start_train.sh \
  --config configs/experiments/fcos3d_exp_pgda.yaml \
  --set experiment.name=experiment_gpu0

CUDA_VISIBLE_DEVICES=1 scripts/start_train.sh \
  --config configs/experiments/fcos3d_exp_pgda.yaml \
  --set experiment.name=experiment_gpu1

scripts/list_train.sh
scripts/stop_train.sh experiment_gpu1
```

`start_train.sh`根据 `CUDA_VISIBLE_DEVICES` 自动选择运行模式：一个设备使用普通
单进程训练，多个设备使用单机 DDP。例如 `CUDA_VISIBLE_DEVICES=0,1` 会启动两个
rank。`data.batch_size_per_gpu`始终表示每张卡的 batch，全局 batch 等于该值乘以
GPU 数量；从单卡 batch 32 切换为双卡且希望保持全局 batch 32 时，应设置每卡
batch 16。训练脚本不会自动放大学习率。

epoch 中间的 `recovery.pth` 只能在相同 world size、每卡 batch 和每 epoch step 数
下精确恢复。改变GPU数量时应使用epoch边界的 `last.pth`；不兼容的中途恢复会在
训练开始前报错，避免样本重复或遗漏。

监督进程名包含 `DetectionTrain:<job-id>`。运行信息按任务保存在
`.runtime/jobs/<job-id>/`，其中包含 `metadata.json`、`pid`、`child.pid`、`status`、
`events.log` 和 `supervisor.log`。无参数调用 `stop_train.sh` 仅在当前恰好有一个任务
时生效；存在多个任务时必须给出实验名或 job-id，避免停错任务。Python
日志库会同时输出数据规模、完整配置、模型参数量、逐步 loss、学习率、显存、验证
指标和 checkpoint 信息到 `outputs/<experiment>/logs/train.log`。前台直接执行
`python3 -u tools/train.py ...` 时，相同内容也会实时显示在终端。

后台启动脚本包含训练监督进程。Python 异常、段错误等导致训练进程退出时，监督
进程会先清理同一进程组内遗留的 DataLoader/CUDA 子进程，等待 30 秒后自动重启，
最多重试 5 次；可通过 `TRAIN_RESTART_DELAY` 和 `TRAIN_MAX_RESTARTS` 环境变量
调整。重启前还会等待旧进程组和旧训练 PID 的 CUDA 上下文释放，默认最多等待
60 秒（可用 `TRAIN_GPU_CLEANUP_TIMEOUT` 调整）；若显存仍属于旧 PID，则停止重启，
避免形成连续 OOM 循环。退出码、信号和重启次数记录在对应任务的 `status` 与
`events.log`。Python
无法捕获的原生崩溃栈会尽可能写入
`outputs/<experiment>/logs/native_crash.log`。

旧版 `.runtime/DetectionTrain.pid` 对应的训练进程不能与新任务注册机制混用；应先
停止旧训练，再通过 `scripts/start_train.sh` 启动受管理任务。

训练默认每 100 个成功 batch 原子覆盖一次 `checkpoints/recovery.pth`，其中包括
模型、优化器、学习率调度器、AMP、epoch/step 和随机状态。自动重启会选择
`recovery.pth` 与 `last.pth` 中较新的文件继续，因此最多重复不足 100 个 batch，
不会从整个 epoch 开头重跑。保存间隔可调整，设置为 0 可关闭：

```bash
--set train.recovery_checkpoint_every_steps=50
```

恢复点是完整训练 checkpoint；R101 模型约需额外 600MB 磁盘空间，写入期间还会
短暂产生同等大小的 `.tmp` 文件。若 CUDA 驱动已经进入持续 Xid 故障状态，有限
次数自动重启仍可能全部失败，此时需要更换 GPU 或由管理员重置设备。

训练默认启用 NaN/Inf 熔断：每步检查 loss 和梯度，每 100 step 以及 checkpoint
保存前检查全部模型参数。发现非有限值时会在优化或保存前终止训练，不覆盖已有的
`last.pth`，并将异常类型、epoch、step、学习率、AMP scale、loss 和异常张量写入
`outputs/<experiment>/diagnostics/nonfinite_*.json`。参数全量检查间隔可以调整：

```bash
--set train.nonfinite_parameter_check_every=50
```

不建议关闭；如仅为诊断兼容性，可设置 `train.nonfinite_guard=false`。
AMP 初始 scale 默认为 `2048`；偶发梯度溢出会跳过当前 optimizer 和 scheduler
更新并自动降低 scale，只有连续 8 次溢出才触发熔断。

启用几何深度、深度传播图和位置相关深度融合的完整 PGD 实验：

```bash
# 本地数据，单卡后台训练
CUDA_VISIBLE_DEVICES=0 scripts/start_train.sh \
  --config configs/experiments/fcos3d_full_pgd_local.yaml

# 远程 ready-v5 数据，单卡后台训练
CUDA_VISIBLE_DEVICES=0 scripts/start_train.sh \
  --config configs/experiments/fcos3d_full_pgd.yaml
```

完整 PGD 使用新的 `conv_geo_weight` 参数，不能严格加载 PGDA checkpoint。若需要
用已有 PGDA 权重初始化，应先通过 checkpoint 转换工具忽略新增分支，并从新的
optimizer 状态开始训练，不能把它作为 `train.resume` 使用。

使用公开 FCOS3D 风格的 ResNet-101 + FPN，并在骨干 C4/C5 和检测 Head 最后一层
启用 modulated DCNv2：

```bash
# 本地数据，单卡后台训练
CUDA_VISIBLE_DEVICES=0 scripts/start_train.sh \
  --config configs/experiments/fcos3d_r101_fpn_dcn_local.yaml

# 远程 ready-v5 数据，单卡后台训练
CUDA_VISIBLE_DEVICES=0 scripts/start_train.sh \
  --config configs/experiments/fcos3d_r101_fpn_dcn.yaml
```

该模型显著大于 EfficientNet-B0 基线，配置默认单卡 batch size 为 2。DCNv2 调用
与当前 PyTorch 匹配的 `torchvision.ops.deform_conv2d`，启动正式训练前建议先用
`runtime.max_train_samples=2 train.epochs=1` 做一次显存冒烟测试。当前配置不自动
下载 ResNet Caffe 预训练权重，因此骨干默认不冻结；接入兼容权重后可设置
`backbone_frozen_stages: 1` 和 `backbone_norm_eval: true` 对齐公开训练配方。

在同一套 ResNet-101 + FPN + DCNv2 网络上启用概率深度、几何深度传播和自适应
融合，使用组合配置：

组合配置通过 `train.pretrain` 指向已经转换为项目键名的 ResNet-101 权重；这是
warm-start，不应配置为 `train.resume`。checkpoint 和数据集的一次性迁移脚本不属于
日常训练、验证工具，不随项目维护；历史脚本仅保留在开发机忽略目录中。

```bash
# 本地 MW3D
CUDA_VISIBLE_DEVICES=0 scripts/start_train.sh \
  --config configs/experiments/fcos3d_r101_fpn_dcn_full_pgd_local.yaml

# 远程 /media/datasets/lzy/mw3d（容器内映射路径）
CUDA_VISIBLE_DEVICES=0 scripts/start_train.sh \
  --config configs/experiments/fcos3d_r101_fpn_dcn_full_pgd.yaml
```

远程宿主机数据目录 `/media/datasets/lzy/mw3d` 在训练容器中挂载为
`/data/horizon_j5/data/lzy/mw3d`；远程 R101 配置使用后一个容器路径。

推理和 ONNX 导出：

```bash
python tools/infer.py --config configs/experiments/fcos3d_exp_pgda.yaml \
  --checkpoint outputs/fcos3d_mw3d_ready_v5_bgr_exp_pgda/checkpoints/best.pth \
  --image path/to/image.jpg --calib path/to/FrontViewCalibParam.json

python tools/export_onnx.py --config configs/experiments/fcos3d_exp_pgda.yaml \
  --checkpoint outputs/fcos3d_mw3d_ready_v5_bgr_exp_pgda/checkpoints/best.pth
```

单图推理需要使用逐帧外参时，额外传入 HAT 格式的
`--extrinsic path/to/frame.txt`；省略时使用标定 JSON 内嵌外参。

详细设计、数据约定和当前兼容边界见 [docs/architecture.md](docs/architecture.md)。

旋转 NMS 固定使用项目内置 CUDA 实现，不依赖 Horizon 插件，也不提供 CPU
推理回退。运行训练、验证和推理时必须保证预测张量位于 CUDA 设备。
