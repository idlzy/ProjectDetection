import re
import json
import matplotlib.pyplot as plt
from pathlib import Path


def parse_hat_train_log(log_path):
    """
    解析 FCOS3D 训练日志（适配实际 train.log 格式）

    日志关键格式:
      训练行:
        epoch=1/100 step=1/1165 global_step=1 lr=... grad_norm=... step_time=... gpu_mem=...
        loss_cls=1.12341 loss_offset=0.20515 loss_depth=2.24028 ... loss_total=6.46889
      验证:
        Validation started | epoch=1
        Validation finished | seconds=98.40 | metrics={"NDS": 0.0533, "mAP": 0.0071, ...}
      best:
        New best checkpoint | NDS=0.053329 | path=...

    返回:
        train_loss: dict[str, list[tuple[int, float]]]
        val_metrics: dict[str, list[tuple[int, float]]]
        best_info: dict {"epoch":int|None, "metric_name":str, "best_val":float|None}
    """
    # ---------正则表达式----------
    # 训练行: epoch=1/100 ... global_step=1 ... loss_cls=1.12 loss_offset=0.2 ...
    # 注意: 只需要 epoch 和 loss_xxx=value 这些
    train_epoch_pat = re.compile(r"epoch=(\d+)/\d+\s+step=")
    loss_item_pat = re.compile(r"(loss_\w+)=([0-9.eE+\-]+)")

    # 验证开始: Validation started | epoch=1
    val_start_pat = re.compile(r"Validation started\s*\|\s*epoch=(\d+)")

    # 验证结束: metrics={...}  这里用宽松匹配拿到 JSON 串
    val_metrics_pat = re.compile(r"metrics=(\{.*\})\s*$")

    # New best checkpoint | NDS=0.053329 | path=...
    best_ckpt_pat = re.compile(r"New best checkpoint\s*\|\s*(\w+)=([0-9.eE+\-]+)")

    # ---------中间容器----------
    # loss: {loss_name: {epoch: last_value_in_that_epoch}}
    # 说明: 一个 epoch 有多个 step 日志, 我们取该 epoch 最后一个 step 的值
    train_loss_raw = {}

    # val: {metric_name: {epoch: value}}
    val_metrics_raw = {}

    # best checkpoint 记录: list of (metric_name, value)
    best_ckpt_list = []

    # 记录最后一次 "Validation started" 的 epoch，用于把 metrics 归到该 epoch
    pending_val_epoch = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        # 1) 解析训练 loss
        m_epoch = train_epoch_pat.search(line)
        if m_epoch and "loss_" in line:
            epoch = int(m_epoch.group(1))
            items = loss_item_pat.findall(line)
            for name, val_str in items:
                try:
                    v = float(val_str)
                except ValueError:
                    continue
                if name not in train_loss_raw:
                    train_loss_raw[name] = {}
                # 后面同 epoch 的覆盖前面的 -> 最终保留该 epoch 最后一个 step 的 loss
                train_loss_raw[name][epoch] = v

        # 2) 记录 Validation started 的 epoch
        m_vs = val_start_pat.search(line)
        if m_vs:
            pending_val_epoch = int(m_vs.group(1))

        # 3) 解析 Validation finished 的 metrics
        m_vm = val_metrics_pat.search(line)
        if m_vm and pending_val_epoch is not None:
            try:
                metrics = json.loads(m_vm.group(1))
            except json.JSONDecodeError:
                metrics = {}
            for fname in ["NDS", "mAP", "mATE", "mASE", "mAOE", "mADE",
                          "mRecall", "mPrecision", "F1"]:
                if fname in metrics:
                    val_metrics_raw.setdefault(fname, {})[pending_val_epoch] = float(metrics[fname])
            pending_val_epoch = None

        # 4) New best checkpoint
        m_best = best_ckpt_pat.search(line)
        if m_best:
            metric_name = m_best.group(1)
            try:
                metric_val = float(m_best.group(2))
            except ValueError:
                metric_val = None
            best_ckpt_list.append((metric_name, metric_val))

    # ---------整理输出----------
    # train_loss -> {name: [(epoch, val), ...]}  按 epoch 排序
    train_loss = {}
    for loss_name, ep_dict in train_loss_raw.items():
        train_loss[loss_name] = sorted(ep_dict.items(), key=lambda x: x[0])

    # val_metrics -> {name: [(epoch, val), ...]}  按 epoch 排序
    val_metrics = {}
    for met_name, ep_dict in val_metrics_raw.items():
        val_metrics[met_name] = sorted(ep_dict.items(), key=lambda x: x[0])

    # ---------构造 best_info----------
    # 优先用日志里的 New best checkpoint 记录
    # 但日志里 New best checkpoint 行没有 epoch 号, 我们用指标+数值反查 epoch
    best_info = {"epoch": None, "metric_name": "unknown", "best_val": None}

    if best_ckpt_list and val_metrics:
        # 用最后一次 best 记录的 metric_name / value 去 val_metrics 里找对应 epoch
        last_metric_name, last_metric_val = best_ckpt_list[-1]
        best_info["metric_name"] = last_metric_name
        best_info["best_val"] = last_metric_val

        # 在对应指标里找最接近该值的 epoch
        metric_series = val_metrics.get(last_metric_name, [])
        if metric_series and last_metric_val is not None:
            best_epoch = None
            best_diff = float("inf")
            for ep, v in metric_series:
                d = abs(v - last_metric_val)
                if d < best_diff:
                    best_diff = d
                    best_epoch = ep
            best_info["epoch"] = best_epoch

    # 如果没能从 best 记录推出 epoch, 回退用 NDS 最大 epoch
    if best_info["epoch"] is None:
        nds_list = val_metrics.get("NDS", [])
        if nds_list:
            best_epoch = max(nds_list, key=lambda x: x[1])[0]
            best_info = {
                "epoch": best_epoch,
                "metric_name": "NDS(max)",
                "best_val": dict(nds_list)[best_epoch],
            }

    # Debug 打印
    print("[DEBUG] 验证集解析条数:")
    for k, v in val_metrics.items():
        print(f"   {k}: {len(v)} records")
    print(f"[DEBUG] New best checkpoint 记录条数: {len(best_ckpt_list)}")
    if best_ckpt_list:
        print(f"[DEBUG] 最后一次 best 记录: {best_ckpt_list[-1]}")

    return train_loss, val_metrics, best_info


def draw_loss_figure(train_loss, out_file="loss_overview.png"):
    """
    图1 loss总览：2行子图
    ax1: 普通子loss (cls / offset / size / yaw / depth_cls / direction / centerness)
    ax2: loss_depth 单独子图（量级太大）
    另外剔除 loss_total 单独不画（或画在 ax1 里也可以，量级接近）
    """
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 10), dpi=120)

    # 深拷贝，避免 pop 影响外面
    train_loss = {k: list(v) for k, v in train_loss.items()}

    # 单独拎出 loss_depth（量级明显更大）
    depth_data = train_loss.pop("loss_depth", None)

    # loss_total 也单独拿出来（可选）
    total_data = train_loss.pop("loss_total", None)

    # 子图1：其余各项loss
    for loss_name, ep_val in train_loss.items():
        eps = [x[0] for x in ep_val]
        vals = [x[1] for x in ep_val]
        ax1.plot(eps, vals, marker=".", linewidth=1.3, label=loss_name)

    # loss_total 用粗虚线叠加
    if total_data is not None:
        eps_t = [x[0] for x in total_data]
        vals_t = [x[1] for x in total_data]
        ax1.plot(eps_t, vals_t, linestyle="--", linewidth=1.8, label="loss_total")

    ax1.set_title("Training Loss (exclude loss_depth)")
    ax1.set_ylabel("Loss Value")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)

    # 子图2 loss_depth
    if depth_data is not None:
        eps_d = [x[0] for x in depth_data]
        vals_d = [x[1] for x in depth_data]
        ax2.plot(eps_d, vals_d, c="#c82423", marker=".", linewidth=1.4, label="loss_depth")
    ax2.set_title("loss_depth curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss Value")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.show()
    print(f"loss曲线图保存: {out_file}")


def draw_model_select_figure(val_metrics, best_info, out_file="model_selection.png"):
    """
    图2：模型选取可视化
    - 绘制NDS、mAP上升曲线；mATE/mASE/mAOE/mADE是误差指标（越小越好）
    - 垂直标记线标出最优epoch
    """
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 10), dpi=120)
    best_ep = best_info["epoch"]
    best_metric = best_info["metric_name"]
    best_val = best_info["best_val"]

    # ax1: 越高越好的指标 NDS, mAP
    has_val_data = False
    for name in ["NDS", "mAP"]:
        ep_val = val_metrics.get(name, [])
        if len(ep_val) > 0:
            eps = [x[0] for x in ep_val]
            vals = [x[1] for x in ep_val]
            ax1.plot(eps, vals, marker="o", markersize=3, linewidth=1.2, label=name)
            has_val_data = True

    if best_ep is not None and best_ep > 0:
        label_text = f"Best ckpt @Epoch {best_ep}"
        if best_val is not None:
            label_text += f"\n({best_metric}={best_val:.4f})"
        ax1.axvline(x=best_ep, linestyle="--", color="#ff4444", linewidth=1.6, label=label_text)

    ax1.set_title("Val Metrics (Higher is better: NDS / mAP)")
    ax1.set_ylabel("Metric Value")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(alpha=0.3)
    if not has_val_data:
        ax1.text(0.5, 0.5, "NO VALID VALIDATION DATA",
                 ha="center", va="center", transform=ax1.transAxes,
                 fontsize=14, color="red")

    # ax2: 误差指标 mATE,mASE,mAOE,mADE，越小越好
    has_err_data = False
    for name in ["mATE", "mASE", "mAOE", "mADE"]:
        ep_val = val_metrics.get(name, [])
        if len(ep_val) > 0:
            eps = [x[0] for x in ep_val]
            vals = [x[1] for x in ep_val]
            ax2.plot(eps, vals, marker="o", markersize=3, linewidth=1.2, label=name)
            has_err_data = True
    if best_ep is not None and best_ep > 0:
        ax2.axvline(x=best_ep, linestyle="--", color="#ff4444", linewidth=1.6)

    ax2.set_title("Val Error Metrics (Lower is better: mATE/mASE/mAOE/mADE)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Error Value")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(alpha=0.3)
    if not has_err_data:
        ax2.text(0.5, 0.5, "NO VALID VALIDATION DATA",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=14, color="red")

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.show()
    print(f"模型选择曲线图保存: {out_file}")
    print(f">>> 最优模型选取信息: {best_info}")


if __name__ == "__main__":
    # ----------------------配置修改----------------------
    LOG_PATH = "outputs/fcos3d_mw3d_exp_pgda_0917/logs/train.log"          # 改成你的日志路径
    OUT_LOSS_IMG = "outputs/train_log/epoch_loss_subplots.png"
    OUT_MODEL_SEL_IMG = "outputs/train_log/val_model_selection.png"
    # ----------------------------------------------------

    train_loss_data, val_metric_data, best_model_meta = parse_hat_train_log(LOG_PATH)

    print("=== 解析结果 ===")
    print("训练loss字段:", list(train_loss_data.keys()))
    for k, v in train_loss_data.items():
        print(f"   {k}: {len(v)} epochs, 范围 {v[0][0]}~{v[-1][0]}" if v else f"   {k}: 空")
    print(f"最优模型信息: {best_model_meta}")

    # 绘图1 loss多子图
    draw_loss_figure(train_loss_data, out_file=OUT_LOSS_IMG)
    # 绘图2 模型选择 + 验证集指标
    draw_model_select_figure(val_metric_data, best_model_meta, out_file=OUT_MODEL_SEL_IMG)