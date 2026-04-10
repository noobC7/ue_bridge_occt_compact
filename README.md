# UE Bridge OCCT Compact

基于AirSim模拟器和OCCT的多智能体强化学习仿真环境，专门用于车队协同控制的研究与开发。

本项目提供了连接AirSim虚拟环境和强化学习训练环境的完整桥梁，支持多车队协同行驶、跟车控制、编队行驶等场景的仿真和算法验证。

## 功能特性

- 🚗 **多车队协同控制**：支持多个车辆在道路上的协同行驶和编队控制
- 🎮 **AirSim仿真接口**：完整的AirSim通信接口，支持车辆状态读取和控制命令发送
- 📍 **道路投影系统**：将车辆位置精确投影到参考道路中心线（Frenet坐标系）
- 👁️ **观察空间构建**：为每个车辆构建包含自身状态、前车/后车状态的观察向量
- 🎛️ **多种控制器**：
  - PID纵向控制器
  - Stanley横向控制器
  - 恒定速度控制器
  - 预训练Actor模型控制器
- 📊 **可视化工具**：支持道路、观察点、车辆轨迹的可视化
- 🧠 **强化学习支持**：集成MAPPO/IPPO算法，支持端到端训练

## 技术栈

- **仿真平台**：AirSim（微软自动驾驶仿真平台）
- **仿真客户端**：IVS (Intelligent Vehicle Simulator)
- **深度学习框架**：PyTorch + TorchRL
- **强化学习算法**：MAPPO（Multi-Agent PPO）/ IPPO（Independent PPO）
- **数值计算**：NumPy
- **配置管理**：PyYAML + Hydra
- **可视化**：Matplotlib
- **通信协议**：msgpackrpc

## 目录结构

```
ue_bridge_occt_compact/
├── airsim_occt_env.py              # 多智能体强化学习环境主类
├── airsim_occt_env_demo.py         # 演示和测试程序
├── airsim_occt_batch_eval.py       # 多方法多道路批量测试脚本
├── airsim_occt_config.py           # 配置管理（环境/车辆/控制/观察）
├── airsim_occt_controllers.py      # 控制器实现
├── airsim_occt_airsim_io.py        # AirSim通信接口
├── airsim_occt_schema.py           # 数据结构定义
├── airsim_occt_shared_obs_core.py  # 观察空间核心
├── airsim_occt_map_projector.py    # 道路投影器
├── airsim_occt_fleet_registry.py   # 车队注册管理
├── airsim_occt_obs_manifest.py     # 观察空间配置
├── airsim_occt_history.py          # 历史状态管理
├── airsim_occt_geometry.py         # 几何计算工具
├── airsim_occt_transform.py        # 坐标转换
├── airsim_occt_plot_actor_log.py   # 统一的日志统计/绘图脚本
├── configs/                        # 配置文件目录
│   └── algorithm/                 # 算法配置
│       ├── default.yaml           # 默认MARL配置
│       ├── actor_baseline.yaml    # MARL/Actor模型基线
│       ├── pid_baseline.yaml      # PID基线
│       ├── mppi_baseline.yaml     # MPPI基线
│       └── constant_baseline.yaml # 恒定控制基线
├── ivs_python_example/            # IVS示例代码
│   ├── mappo_ippo_occt.py        # MAPPO/IPPO训练算法
│   ├── occt_map.py               # OCCT地图处理
│   └── occt_scenario.py          # OCCT场景定义
├── setup_vsim.py                  # IVS模块路径设置
├── path_visualize.py              # 路径可视化工具
└── README.md                      # 本文件
```

## 核心模块说明

### AirSimOcctMARLEnv (`airsim_occt_env.py`)
多智能体强化学习环境主类，管理车辆注册、观察空间构建、控制命令发送，支持重置、步进、渲染等标准gym-like接口。

### 配置系统 (`airsim_occt_config.py`)
- `EnvConfig`：环境配置（地图路径、车辆数量等）
- `VehicleConfig`：车辆配置（初始位置、速度等）
- `ControlConfig`：控制参数（PID增益、Stanley参数等）
- `ObsConfig`：观察空间配置（观察维度、邻居信息等）

### 控制器 (`airsim_occt_controllers.py`)
- `LowLevelController`：底层控制器（油门、刹车、转向）
- `PIDLongitudinalController`：纵向速度PID控制
- `SteeringAdapter`：转向控制适配器
- `ConstantController`：恒定速度控制
- `ActorModelController`：预训练模型控制
- `CenterlinePIDController`：中心线跟踪PID/Stanley基线
- `CenterlineMPPIController`：中心线跟踪MPPI基线

### AirSim接口 (`airsim_occt_airsim_io.py`)
处理与AirSim的通信，包括车辆状态读取、控制命令发送、坐标转换和数据同步。

### 道路投影器 (`airsim_occt_map_projector.py`)
将车辆位置投影到道路中心线，支持Frenet坐标系转换（s, y坐标）和道路曲率计算。

## 快速开始

### 运行环境

```bash
conda activate /home/yons/Graduation/pyquaticus/env-full
cd /home/yons/Graduation/ue_bridge_occt_compact
```

当前默认配置 `configs/algorithm/default.yaml` 使用最新的 MARL checkpoint：

```text
/home/yons/Graduation/rl_occt/outputs/2026-04-04/19-14-19_mlp_ippo_train/checkpoints/checkpoint_iter_299_frames_18000000.pt
```

### 连通性自检

在启动 UE/AirSim 后，先检查车辆状态是否能正常读取：

```bash
python airsim_occt_smoke_test.py \
  --host 127.0.0.1 \
  --port 41451 \
  --vehicles vehicle0 vehicle1 vehicle2 vehicle3 vehicle4 \
  --count 3
```

### 单次运行

说明：
- `--plot-road` 现在默认开启，若不想在 AirSim 中画道路，可显式传 `--no-plot-road`
- `--show-log` 默认关闭，只在需要逐步打印 step 日志时开启
- `--show-render-time` 默认关闭，只在需要统计 AirSim 绘制耗时时开启
- `--step-count` 默认是 `2000`
- 当前单次 run 在以下任一条件满足时结束并保存：
  - 达到最大步长 `step-count`
  - 触发终点 `goal_reached`

#### MARL

```bash
python airsim_occt_env_demo.py \
  --algo-config configs/algorithm/default.yaml \
  --map-dir /home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path \
  --road-env-index 1 \
  --vehicles vehicle0 vehicle1 vehicle2 vehicle3 vehicle4 \
  --step-count 2000 \
  --output-suffix marl_road1
```

如果希望查看 MARL 的实时特色可视化，可增加：

```bash
--plot-marl-debug
```

该模式会在 AirSim 中为中间三辆 follower 实时绘制：
- 感知到的左右边界点
- `ref_short_term` 参考轨迹点
- Actor 输出动作箭头
  - 曲率由转向角计算
  - 颜色表示加速度（红色减速，绿色加速）

#### PID

```bash
python airsim_occt_env_demo.py \
  --algo-config configs/algorithm/pid_baseline.yaml \
  --map-dir /home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path \
  --road-env-index 1 \
  --vehicles vehicle0 vehicle1 vehicle2 vehicle3 vehicle4 \
  --step-count 2000 \
  --output-suffix pid_road1
```

当前部署侧 PID 纵向控制已改成**弧长位置反馈**，不再只是单纯跟踪前车速度：

```yaml
controller:
  pid:
    platoon_position_gain: 0.8
```

#### MPPI

```bash
python airsim_occt_env_demo.py \
  --algo-config configs/algorithm/mppi_baseline.yaml \
  --map-dir /home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path \
  --road-env-index 1 \
  --vehicles vehicle0 vehicle1 vehicle2 vehicle3 vehicle4 \
  --step-count 2000 \
  --output-suffix mppi_road1
```

如果希望查看 MPPI 的实时规划调试曲线，可增加：

```bash
--plot-mppi-debug
```

该模式会在 AirSim 中为每个 follower 实时绘制：
- `ref_points`
- `sampled_trajs`
- `optimal_traj`

当前 `mppi_baseline.yaml` 已将 horizon 调整为更接近 `occt_scenario.py` 的设置：

```yaml
controller:
  mppi:
    horizon_steps: 30
    num_samples: 256
    lambda: 10.0
    exploration: 0.1
```

### 日志命名

```text
tracking_YYYYMMDD_HHMMSS_<method>_roadX
```

例如：

```text
tracking_20260408_220104_pid_road0
```

当 `--repeats > 1` 时，同一个 `method-road` 组合会生成一个目录，目录内保存多个独立 run 的日志文件：

```text
tracking_20260409_120000_pid_road0/
├── tracking_log_0.json
├── tracking_log_1.json
└── tracking_log_2.json
```

## 批量测试

对 `PID / MPPI / MARL` 三种方法在 `road0..5` 上批量测试：

```bash
python airsim_occt_batch_eval.py \
  --methods pid mppi marl \
  --roads 0 1 2 3 4 5 \
  --map-dir /home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path \
  --vehicles vehicle0 vehicle1 vehicle2 vehicle3 vehicle4 \
  --step-count 2000 \
  --output-dir /home/yons/Graduation/ue_bridge_occt_compact/airsim_occt_tracking_outputs/0410
```

当前批量测试的默认行为：
- `--plot-road` 默认开启
- `--plot-marl-debug` 默认关闭
- `--plot-mppi-debug` 默认关闭
- `--show-log` 默认关闭

如果希望每个 `method-road` 组合重复多次独立运行：

```bash
python airsim_occt_batch_eval.py \
  --methods pid mppi marl \
  --roads 0 1 2 3 4 5 \
  --repeats 3 \
  --map-dir /home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path \
  --vehicles vehicle0 vehicle1 vehicle2 vehicle3 vehicle4 \
  --step-count 2000 \
  --output-dir /home/yons/Graduation/ue_bridge_occt_compact/airsim_occt_tracking_outputs
```

`--repeats` 的语义是：
- 同一个 `method-road` 组合对应一个输出目录
- 每个独立 run 会单独保存一个 `tracking_log_i.json`
- 每个独立 run 在达到终点或超过最大步长后立即结束并保存

如果希望某条路失败后继续后面的组合，追加：

```bash
--continue-on-error
```

## 日志统计与绘图

统一使用一个脚本处理：
- 单个 tracking log 的时序图
- 整个 `airsim_occt_tracking_outputs/` 根目录下的 CSV 指标统计

注意：
- 当前统计/绘图脚本**只支持新日志格式**
- 如果日志缺少以下字段，会直接报错，不再兼容旧日志：
  - `target_agent_s`
  - `distance_to_ref`
  - `closest_center_map`
  - `hinge_target_speed`
- 这样做是为了避免兼容逻辑掩盖真实问题；重新跑测试后再做统计与绘图
- `hinge_distance` 是唯一例外：
  - 新日志若已写入则直接使用
  - 像 `0410` 这类未落盘该字段的日志，会在绘图时根据 `pose_map_xy` 和首尾车 `closest_center_map` 现场重建

### 单个 log 绘图

```bash
python airsim_occt_plot_actor_log.py \
  --log-file /path/to/tracking_log.json
```

对于单个 log：
- `PID` 会绘制中间三车的 `pid_timeseries`
- `MPPI` 会绘制中间三车的 `mppi_timeseries`
- `MARL` 会绘制中间三车的 `marl_timeseries`
- 还会额外生成：
  - `hinge_state_timeline.pdf`
  - `controller_compute_time.pdf`
  - `platoon_longitudinal_error.pdf`
  - `platoon_lateral_error.pdf`
  - `group_trajectory.pdf`
  - `group_target_acc.pdf`
  - `group_speed.pdf`
  - `group_policy_delta.pdf`

对于 `MARL / MPPI` 的 steering 子图：
- 现在只绘制两条曲线：
  - 算法输出角（`policy_delta` / `command_delta`）
  - 根据控制参数在绘图时重建的一阶惯性估计值（`estimated_delta`）
- `estimated_delta` 不再存储在日志中，而是绘图时动态计算

### 根目录批量出表

```bash
python airsim_occt_plot_actor_log.py \
  --log-file /home/yons/Graduation/ue_bridge_occt_compact/airsim_occt_tracking_outputs/0410 \
  --csv-only
```

会在根目录下生成：
- `tracking_metrics_runs.csv`
- `tracking_metrics_summary.csv`
- `tracking_metrics_roundabout.csv`
- `tracking_metrics_right_angle_turn.csv`
- `tracking_metrics_s_curve.csv`
- `tracking_metrics_overall.csv`

脚本会自动扫描以下文件：
- `tracking_log.json`
- `tracking_log_*.json`

### 根目录批量出表并给每个 run 生成图

```bash
python airsim_occt_plot_actor_log.py \
  --log-file /home/yons/Graduation/ue_bridge_occt_compact/airsim_occt_tracking_outputs/0410 \
  --generate-plots
```

当前 CSV 指标会统计一批与上游 `occt_metrics_evaluation.py` 命名风格接近的指标，例如：
- `s_error_mean / std / max`
- `ttc_global_min`
- `acc_mean / std / max`
- `jerk_mean / std / max`
- `ste_rate_mean / std / max`
- `hinge_time`
- `hinge_count`
- `hinge_spe_diff`
- `hinge_ratio_mean`
- `hinge_ready_ratio_mean`
- `occt_ratio_mean`
- `controller_compute_time_ms_mean / std / max / total`

注意：
- `controller_compute_time_ms` 是**每一步控制器计算耗时**
- 它记录在 `tracking_log.json` 的每一步 `info` 中
- 适用于比较 `PID / MPPI / MARL` 三种算法的实时性

输出图保存在对应 tracking 目录下的 `plots/` 子目录；CSV 直接生成在指定根目录下。

各 CSV 的含义：
- `tracking_metrics_runs.csv`
  - 每个独立 run 一行
- `tracking_metrics_summary.csv`
  - 按 `method + road_id` 聚合，一条道路上每种方法一行
- `tracking_metrics_roundabout.csv`
  - 将 `road0` 和 `road1` 按场景类型 `roundabout` 聚合，每种方法一行
- `tracking_metrics_right_angle_turn.csv`
  - 将 `road2` 和 `road3` 按场景类型 `right_angle_turn` 聚合，每种方法一行
- `tracking_metrics_s_curve.csv`
  - 将 `road4` 和 `road5` 按场景类型 `s_curve` 聚合，每种方法一行
- `tracking_metrics_overall.csv`
  - 将所有道路整体聚合，每种方法一行

### 论文分析相关绘图

对于每个单独 run，脚本会额外生成：
- `platoon_longitudinal_error.pdf`
  - 三个 follower 的编队纵向误差曲线
  - 背景透明色标注各 follower 的 `hinge_ready_status=True` 时段
- `platoon_lateral_error.pdf`
  - 三个 follower 的横向误差曲线
  - 背景透明色同样标注 `hinge_ready_status=True` 时段
- `controller_compute_time.pdf`
  - 当前 run 的控制器计算时间曲线
- `hinge_state_timeline.pdf`
  - 仅显示中间三辆 follower
  - 中文图例：`可铰接`、`铰接点距离`、`铰接完成`
  - `可铰接` 以背景透明色块表示，且阴影只覆盖 `y∈[0,1]`
  - 左轴绘制 `铰接点距离 (m)`，右轴绘制 `铰接完成 (0/1)`，两侧刻度颜色与对应曲线一致
  - 如果日志里没有 `hinge_distance`，脚本会按首尾车中心线插值位置离线重建后再绘制
  - `铰接完成` 曲线从首次满足 `hinge_ready_status && occt_state` 的时刻开始保持为 1
- `group_trajectory.pdf`
  - 三个 follower 的轨迹对比图
  - 会叠加当前道路的左右边界，左边界为蓝色、右边界为红色
  - 对 `road3 / road4 / road5` 会自动交换 `x/y` 并按画布宽高比重设坐标范围，以避免长条形轨迹图
  - 主轨迹线会使用更细的线宽，减轻重叠遮挡
  - `road0` 到 `road5` 使用固定的局部放大框配置，不再自动选点
  - 每条路都会按预设的源矩形区域、放大框中心位置和 `2x` 放大倍率生成一个或多个 inset，且 inset 不显示坐标刻度
  - 源框只保留矩形标注，不再绘制源框到 inset 的连接线
- `group_target_acc.pdf`
  - 三个 follower 的目标加速度对比图（PID 使用重建的加速度）
- `group_speed.pdf`
  - 三个 follower 的速度对比图
- `group_policy_delta.pdf`
  - 三个 follower 的目标转角对比图（PID 使用 `command_delta`）

当对整个根目录执行 `--generate-plots` 时，还会在根目录额外生成：
- `controller_compute_time_boxplot.pdf`
  - 所有场景下三种方法控制器计算时间的箱线图对比
  - 使用对数纵轴，便于展示 MPPI 与 MARL/PID 的数量级差异

## 训练和部署

训练相关脚本仍保留在 `ivs_python_example/` 与上游 `rl_occt/` 中；本仓库当前重点是 AirSim/IVS 部署桥接、日志记录和多方法道路测试。

## 项目特点

- ✅ 完整的仿真到训练流程
- ✅ 模块化设计，易于扩展
- ✅ 支持多种控制策略
- ✅ 丰富的可视化工具
- ✅ 兼容gym-like接口
- ✅ 支持多智能体协作

## 应用场景

- 自动驾驶车队控制研究
- 强化学习算法验证
- 智能交通系统仿真
- 车队协同控制算法开发
- 编队行驶策略研究

## 许可证和贡献

本项目仅供学术研究使用。如有问题或建议，欢迎提Issue。
