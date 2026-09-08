# 验证记录

2026-09-08 在远程容器 `lzy_mw3d_fcos3d_j5_v1160` 中完成：

- Python 3.8 / PyTorch 1.13 下安装成功。
- 单元测试通过。
- MW3D v5 split 数量为 25847 / 4230 / 1432，跨 split 的 sample token 和
  split group 重叠均为 0。
- 真实训练样本可读取，首个样本得到 25 个有效 3D target。
- 五层 Head 输出形状为 64×112、32×56、16×28、8×14、4×7。
- FCOS3D + PGDA loss 前向、反向传播通过。
- 单样本训练、validation、best/last checkpoint 保存通过。
- validate 和 test CLI 通过。
- 图片推理与可视化文件保存通过。
- ONNX 导出通过，30 个输出，ONNX checker 通过。
- Horizon CUDA rotated NMS 已接入 `auto` 后端；2,000 个输入框预热后核心算子
  约 1.5 ms，单张真实验证样本完整进程约 3.1 s（含模型与进程启动）。

这些结果证明工程纵向链路可执行，不代表随机初始化 smoke checkpoint 具有检测
精度，也不代表已经与 HAT checkpoint 数值对齐。
