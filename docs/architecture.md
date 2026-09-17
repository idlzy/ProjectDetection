# 架构与兼容边界

## 数据流

`Mw3dReadyDataset` 从 `splits/<split>_frames.jsonl` 获取样本索引，从独立的
`data_root` 读取图片、标注和标定。图片保持 BGR，等比例缩放并补边到
512×896，随后执行 `(x-128)/128`。

训练数据返回 2D 框、投影中心、相机坐标 3D 框、类别以及完整样本元数据。
验证和测试使用同一套解码器与指标实现，避免训练回调和离线测试行为漂移。

## 模型

默认网络采用无 SE、ReLU 的 EfficientNet-B0、三层轻量 BiFPN 和五尺度
FCOS3D Head。EfficientNet 保留 3×3/5×5 深度卷积以及线性残差输出；BiFPN 使用
双向求和融合和深度可分离卷积。Head 跨尺度共享卷积权重、为每个尺度保留独立
归一化，并把中心偏移、深度、尺寸和旋转拆分到独立回归分支。数据集没有可靠的
速度监督，因此普通模型不再预测速度；后处理为统一9维评估接口补零速度。Head 还输出
分类、方向、属性、centerness，以及可选的概率深度和几何深度融合权重。

模型代码按组件职责组织，避免后续加入新骨干或检测器时继续堆积在同一目录：

```text
models/
├── backbones/              # 特征提取网络；新增骨干在这里实现并导出
│   ├── efficientnet.py
│   └── resnet.py           # FCOS3D Caffe-style ResNet-101，可选 stage DCNv2
├── necks/                  # 多尺度特征融合
│   ├── bifpn.py
│   └── fpn.py              # ResNet C3--C5 到 P3--P7
├── heads/                  # 检测头及其输出分支
│   └── fcos3d_head.py
├── detectors/              # 组装 backbone、neck、head 的完整网络
│   └── fcos3d.py
├── layers/                 # 模型内部复用的基础层
│   └── conv.py
├── builder.py              # 配置到完整模型的唯一构建入口
└── __init__.py             # 对外公开 FCOS3D 和 build_model
```

训练、验证、推理和导出只依赖 `project_detection.models.build_model`；组件内部通过
各子包的 `__init__.py` 暴露稳定接口。完整检测器仍以 `backbone`、`neck`、`head`
作为成员名。默认 EfficientNet、BiFPN 和 Head 在轻量化升级后参数键已经变化，
升级前普通模型的 checkpoint 不能对升级后的默认模型执行严格加载；专用 Legacy
模型及其历史 HAT checkpoint 兼容关系不受影响。

`fcos3d_r101_fpn_dcn.yaml` 提供另一条不影响现有默认配置的模型路径：Caffe stride
风格 ResNet-101 输出 C2--C5，FPN 跳过 C2、融合 C3--C5，并从 P5 继续生成 P6、
P7。配置在 ResNet C4/C5 的所有 bottleneck 3×3 卷积及分类/回归 tower 的最后一层
启用 modulated DCNv2，Head 使用 GroupNorm(32)。DCNv2 的 offset/mask 预测器零
初始化，实际算子由 `torchvision.ops.deform_conv2d` 提供。

公开 FCOS3D 配方还使用 Caffe ImageNet 预训练权重、冻结 stem/C2 并固定 BN 统计；
本仓库暂未附带或自动下载该权重，所以新配置用 `backbone_frozen_stages: -1`、
`backbone_norm_eval: false` 从头训练。模块已支持切换为 `1/true`，但只有在加载兼容
预训练权重后才建议如此设置。

PGDA 兼容模式使用：

```text
DR = exp(r)
DP = sum(softmax(logits) * depth_centers)
w = sigmoid(depth_fuse_logit)
depth = w * DR + (1 - w) * DP
```

概率深度同时接受最近深度 bin 的交叉熵监督，以及融合结果的连续深度回归监督。

完整 PGD 使用独立的 `fcos3d_full_pgd.yaml` 配置。在局部深度 `DL` 基础上，先把
五层 FPN 的同图候选汇总为实例节点，再由 `task/depth_propagation.py` 根据去畸变
投影中心、预测高度、深度置信度和分类相似度建立有向图：

```text
DR + DP -> DL
DL + perspective graph -> DG
D = sigmoid(alpha) * DL + (1 - sigmoid(alpha)) * DG
```

`alpha` 是 Head 从每个位置预测的融合权重。DG 没有可学习参数，传播输入全部
detach；没有可靠边、位于地平线附近、不在几何类别白名单或超过图节点上限的候选
回退到 DL。训练中同一 GT 对应的多个正样本不会互相传播，推理中先跨尺度构图、
融合深度，再恢复相机坐标和执行 rotated NMS。

MW3D 保留镜头畸变，因此传播前先把投影中心转换成归一化针孔射线；地平线由
camera-to-vehicle 旋转和车辆坐标系的地面法向量计算。当前实现是面向 MW3D 的
论文公式实现，不声称与未公开的 HAT 几何图算子逐元素一致。

ONNX 导出包含 `geo_weight` 原始特征图。候选筛选、变长图构建和 DG 融合仍属于
部署后处理，不包含在第一阶段的 raw-head ONNX 图中。

## 当前阶段

仓库已具备独立训练、验证、测试、推理和 ONNX 导出入口。第一阶段目标是建立
可运行纵向闭环；HAT 权重键名映射和中间特征 golden data
属于下一阶段的数值对齐工作。当前实现不会声称能无损加载 HAT checkpoint。

后处理提供 `auto/horizon/reference` 三种选择。`auto` 在 CUDA Tensor 且容器安装
了 `horizon_plugin_pytorch` 时调用 `torch.ops.horizon.nms_rotated`，否则回退到
不依赖外部算子的 CPU reference 实现。Horizon 后端用于当前迁移期加速验证；
reference 后端用于独立运行和正确性测试。

## 产物约定

每个实验写入 `outputs/<experiment>/`：

```text
resolved_config.yaml
checkpoints/last.pth
checkpoints/best.pth
metrics/val.json
metrics/test.json
predictions/*.json
visualizations/*
logs/*
```

模型选择只能使用 val；test 只用于最终报告。
