# 训练开机自动恢复（启动时自动登记、保守模式）

## 边界

功能解决“服务器已经重启，但训练没有自动拉起”，不修复内核 Oops，不自动重启
服务器，也不执行 GPU reset。start_train.sh 默认自动登记本次受管任务，不扫描 outputs，
也不会复活未登记的历史训练。训练参数和命令保持原样，不再要求每个实验手动安装服务。
系统重启后从最新有效的 recovery.pth / last.pth 续训，不保证恢复到崩溃前最后一步。

服务使用 systemd **用户服务**，以训练用户运行。必须开启 lingering，才能在无人登录时
启动。以训练用户运行命令，不要用 sudo 执行训练脚本。自动启用用户服务不会额外
启动第二份训练；只为下一次系统启动启用恢复。

## 使用（在训练服务器上执行）

先部署本次更新的 tools、scripts，并确保 ai 环境中的 project_detection 与工程兼容。
以下命令以服务器一的 ziyangl 用户为例：

```bash
# 每台服务器、每个训练用户仅准备一次（需要管理员权限）。
sudo loginctl enable-linger ziyangl
cd /home/ziyangl/Project/ProjectDetection
conda activate ai

# 和以前完全相同；保存快照、启用服务、登记恢复由启动器自动完成。
CUDA_VISIBLE_DEVICES=0 bash scripts/start_train.sh \
  --config configs/experiments/你的配置.yaml
```

启动时应看到 `Boot recovery enabled | job=...`。默认流程先检查用户 systemd 和
lingering，再校验训练配置、预留任务、安装/复用服务并登记，成功后才启动训练。
没有权限或安装失败时不会静默降级：脚本报错且不启动训练。普通脚本不能绕过系统
权限自动执行 sudo。如果暂时不需要开机续训（或运行在不支持 systemd/GPU UUID 的环境），
可以明确选择旧行为：

```bash
TRAIN_BOOT_RESUME=0 CUDA_VISIBLE_DEVICES=0 bash scripts/start_train.sh \
  --config configs/experiments/你的配置.yaml
```

同一实验再次手动启动时，会自动更新快照和登记，并重置恢复次数；已有且内容一致的
服务文件会复用，不要求手工删除/安装。已有服务文件被人为修改或为符号链接时会拒绝
覆盖，需先由管理员检查。不保留独立的手动安装脚本或“仅登记、不启用服务”的旧入口。
启动训练和建立恢复资格统一通过 start_train.sh 完成。

已经运行的旧训练不会因为同步脚本而自动接管。若没有
`.runtime/jobs/<job-id>/launch_config.json`，应在合适窗口用新版启动器明确续训一次；
该次启动会自动登记。不能凭后来修改过的配置猜测原启动配置。
直接使用 `python tools/train.py` 启动的任务不包含此自动登记流程。

脚本会打印完整 job-id；服务名称为 `projectdetection-<job-id>.service`。
每个实验一个服务，默认等待依赖最多 300 秒。不要手工 `systemctl start` 来测试续训：
同一次开机已登记/尝试过的任务会直接跳过，防止重复启动。

## 查看、停止、重新登记

```bash
bash scripts/list_train.sh
systemctl --user status projectdetection-<job-id>.service
journalctl --user -u projectdetection-<job-id>.service -b --no-pager
python tools/train_boot_resume.py status --job-dir .runtime/jobs/<job-id>

# 先持久化取消自动恢复，再停止训练；即使任务已不在线，也可取消资格。
bash scripts/stop_train.sh 你的实验名称

# 完全移除开机启动资格：先执行上面的 stop，再禁用用户服务。
systemctl --user disable --now projectdetection-<job-id>.service
```

只执行 `systemctl --user stop` 被视为服务关闭，不等同于取消续训意图；下次开机仍可能
恢复。用户主动停训请统一使用 stop_train.sh。系统关机发送的 SIGTERM 不会取消资格。
训练正常完成会自动取消资格；保留但禁用实际工作的已启用 unit 是无害的。

手动重新执行 start_train.sh 会先取消同名旧登记，再自动登记本次启动的配置；
设置 TRAIN_BOOT_RESUME=0 时则只取消旧登记，不建立新登记。

若恢复次数用尽，在检查失败原因后通过正常启动入口重新续训，同时自动更新登记和重置预算：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/start_train.sh \
  --config outputs/你的实验名称/resolved_config.yaml --auto-resume
```

任务仍在运行时先通过 stop_train.sh 正常停止，不要同时启动第二份。需要更改配置时，
同样通过启动器重新启动任务（自动登记新的启动快照），而不是编辑 boot_config.json。

## 安全机制

- 开机恢复记录独立存储于 `.runtime/jobs/<job-id>/boot_resume.json`，包括冻结配置摘要、
  Python 路径、物理 GPU UUID、必要环境变量、文件系统挂载点/UUID、恢复意图和次数。
  不记录任意环境变量或凭据；不支持 MIG。索引变动时仍使用原物理 GPU。
- 任务登记使用 boot ID + PID 启动时间，避免跨重启误认 PID；保留项目级任务/GPU互斥。
- tools/train.py 对输出目录增加进程生命周期文件锁，阻止绕过登记的重复 CLI 训练。
  所有参与者必须使用更新后的入口；旧代码、其他工程、其他主机无法由此完全约束。
- 服务在前台管理 supervisor，而不是把 nohup 后台脚本作为 systemd 主进程。
  系统关闭时转发信号；超时后 systemd 终止本服务 cgroup 内的全部进程。
- 检查数据目录、train/val 清单、输出文件系统身份、教师检查点及 GPU 可用性。
  不就绪时限时等待；校验失败则退出并留下记录，不抢占其他 GPU 任务。
- 配置使用启动时快照，并检测冻结配置被修改。CPU/驱动/GPU、Python 包仍需管理员维护；
  快照不会冻结 wheel 或驱动二进制，升级前应先停训并取消恢复。
- 只读取受信任本地 checkpoint（torch.load 不是不可信文件的安全沙箱）。检查训练状态、
  配置兼容性与中途续训的 GPU/batch 拓扑；按 epoch/step 选择，不依赖文件修改时间。
  不可读、不完整或不兼容时尝试另一个检查点；均无效则拒绝从头训练。
  这不是文件内容校验和，也无法识别所有硬件造成的静默数据损坏；实际严格加载和
  训练期检查仍可能失败。自动续训也沿用这一保守选择逻辑。
- 每个任务每次系统启动最多尝试一次，登记所在的当前启动不尝试；默认跨启动最多
  3 次恢复，人工检查后通过启动器重新启动任务才能重置。开机恢复本身绝不会
  重新登记/重置预算。每次开机恢复中的 supervisor
  最多额外重试 2 次；systemd 配置 Restart=no，不叠加无限重试。
- 仅对低频启动快照、恢复意图与次数记录执行 fsync；未改动模型检查点保存方式。
  这保证正常文件系统上的持久化顺序，不保证故障硬件绝不丢数据，也不是 Oops 修复。

## 验证与上线

本地 CPU 回归覆盖检查点排序/损坏回退、拒绝从头训练、跨 boot ID 检查、预算、取消与
完成状态、依赖检查失败、关闭信号、前台命令及互斥。它不能代替实际服务器的开机验收。

上线前使用单独实验、少量样本和短训练：

1. 完成一次正常训练，确认下次服务运行不恢复。
2. 手动 stop，确认取消资格且不会复活。
3. 在安排好的窗口、无其他关键任务时正常重启，确认无需登录便能续训。
4. 使用临时实验验证最新 checkpoint 不可读时回退，全部无效时不从头训练。
5. 确认数据盘缺失/GPU冲突时不启动、同实验第二次启动被拒绝。

测试不自动执行服务器重启或破坏 checkpoint；这些验收步骤需要管理员在维护窗口执行。
正常启动命令会安装/启用相应用户服务，但不会自动重启机器、重置 GPU 或执行 sudo。
清理 `.runtime` 或移动工程目录前必须取消服务，否则登记会失效。
