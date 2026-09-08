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

# 远程 ready-v5
CUDA_VISIBLE_DEVICES=0 scripts/start_train.sh \
  --config configs/experiments/fcos3d_r101_fpn_dcn_full_pgd.yaml
```

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
