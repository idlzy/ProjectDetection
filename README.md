# ProjectDetection

独立 PyTorch 实现的 MW3D 单目 3D 检测工程，覆盖数据检查、FCOS3D/PGDA
建模、训练、验证、测试、推理和 ONNX 导出。运行时不依赖 HAT 或
`horizon_plugin_pytorch`。

当前实现是可运行的第一阶段基线。网络、数据流和命令行已经形成闭环；与
HAT checkpoint 的逐层参数映射和数值对齐放在 `tests/parity` 中继续推进。

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

python3 tools/inspect_dataset.py \
  --config configs/experiments/fcos3d_exp_pgda_local.yaml
```

脚本读取根目录的 `frames.jsonl`，输出 `splits/train_frames.jsonl`、
`val_frames.jsonl` 和 `test_frames.jsonl`。它先按 `image_rel` 中的
`images/<大类目录>/...` 分层，再在每个大类内部以完整 `split_group` 为单位尽量
按 8:1:1 划分。因此每类数据都会尽量覆盖三个集合，同时不会将同一段连续采集
数据拆进不同集合。每个大类的实际帧数和组数记录在 `split_summary.json`；若输出
已经存在，只有显式添加 `--force` 才会覆盖。

先做数据检查和小样本冒烟训练：

```bash
python tools/inspect_dataset.py --config configs/experiments/fcos3d_exp_pgda.yaml

torchrun --standalone --nproc_per_node=2 tools/train.py \
  --config configs/experiments/fcos3d_exp_pgda.yaml \
  --set runtime.max_train_samples=32 runtime.max_val_samples=16 train.epochs=1
```

完整训练、验证和测试：

```bash
torchrun --standalone --nproc_per_node=2 tools/train.py \
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

脚本会自动选择包含 PyTorch、OpenCV 和 PyYAML 的 Python；也可以通过
`TEST_PYTHON=/path/to/python` 显式指定环境。

结果默认写入 `outputs/test_runs/mw3d_test_<时间>/`。可视化目录与参考脚本一致，
按 test manifest 的 `split_group` 分组，图片命名为
`visualizations/<split_group>/<原图stem>_pred.jpg`；此外还包括 `metrics.json`
以及 `plots/` 下的汇总柱状图、分类 AP 图和距离阈值 AP 矩阵。展示阈值默认
为 `0.35`，完整评估阈值默认采用 `0.05`；可分别通过 `VIZ_SCORE_THR` 和
`EVAL_SCORE_THR` 调整。`NMS_BACKEND` 默认为 `auto`：CUDA 环境优先使用
Horizon 算子，其次使用项目内置的精确 CUDA 旋转 NMS，CPU 环境回退到 reference
实现；也可显式设为 `horizon`、`cuda` 或 `reference`。设置
`EVAL_MAX_SAMPLES` 可进行小规模冒烟测试。
可视化默认覆盖完整 test 清单，设置 `VIZ_MAX_IMAGES` 可限量；已有图片默认跳过，
`VIZ_START_INDEX` 可从指定 manifest 下标续跑，`VIZ_OVERWRITE=1` 可覆盖，
`SAVE_BEV=1` 可额外输出 BEV 图。

单卡后台训练使用内置启停脚本，不需要把终端输出重定向到日志文件：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/start_train.sh \
  --config configs/experiments/fcos3d_exp_pgda.yaml

scripts/stop_train.sh
```

训练进程名为 `DetectionTrain`，PID 保存在 `.runtime/DetectionTrain.pid`。Python
日志库会同时输出数据规模、完整配置、模型参数量、逐步 loss、学习率、显存、验证
指标和 checkpoint 信息到 `outputs/<experiment>/logs/train.log`。前台直接执行
`python3 -u tools/train.py ...` 时，相同内容也会实时显示在终端。

后台启动脚本包含训练监督进程。Python 异常、段错误等导致训练进程退出时，监督
进程会等待 30 秒并自动重启，最多重试 5 次；可通过 `TRAIN_RESTART_DELAY` 和
`TRAIN_MAX_RESTARTS` 环境变量调整。退出码、信号和重启次数记录在
`.runtime/DetectionTrain.status` 与 `.runtime/DetectionTrain.events.log`。Python
无法捕获的原生崩溃栈会尽可能写入
`outputs/<experiment>/logs/native_crash.log`。

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

首次运行前，将下载的官方 ResNet-101 Caffe 权重转换为项目键名：

```bash
conda run -n ai python tools/convert_checkpoint.py \
  --config configs/experiments/fcos3d_r101_fpn_dcn_full_pgd.yaml \
  --input checkpoints/pretrained/resnet101_msra-6cc46731.pth \
  --output checkpoints/pretrained/resnet101_msra_project_detection.pth \
  --allow-partial
```

转换报告写入同目录的 `.pth.report.json`。组合配置已通过 `train.pretrain` 指向转换
后的文件；这是 warm-start，不应配置为 `train.resume`。

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

详细设计、数据约定和当前兼容边界见 [docs/architecture.md](docs/architecture.md)。

当前 Horizon 容器中，配置项 `evaluation.nms_backend: auto` 会自动使用插件提供的
CUDA rotated NMS。若要强制检查独立 CPU 结果，可运行时添加：

```bash
--set evaluation.nms_backend=reference
```
