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
├── airsim_occt_plot_actor_log.py   # 日志可视化工具
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

#### MARL

```bash
python airsim_occt_env_demo.py \
  --algo-config configs/algorithm/default.yaml \
  --map-dir /home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path \
  --road-env-index 1 \
  --vehicles vehicle0 vehicle1 vehicle2 vehicle3 vehicle4 \
  --step-count 200 \
  --plot-road \
  --output-suffix marl_road1
```

#### PID

```bash
python airsim_occt_env_demo.py \
  --algo-config configs/algorithm/pid_baseline.yaml \
  --map-dir /home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path \
  --road-env-index 1 \
  --vehicles vehicle0 vehicle1 vehicle2 vehicle3 vehicle4 \
  --step-count 200 \
  --plot-road \
  --output-suffix pid_road1
```

#### MPPI

```bash
python airsim_occt_env_demo.py \
  --algo-config configs/algorithm/mppi_baseline.yaml \
  --map-dir /home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path \
  --road-env-index 1 \
  --vehicles vehicle0 vehicle1 vehicle2 vehicle3 vehicle4 \
  --step-count 200 \
  --plot-road \
  --output-suffix mppi_road1
```

### 日志命名

```yaml
tracking_YYYYMMDD_HHMMSS_<method>_roadX
```

例如：

```text
tracking_20260408_220104_pid_road0
```

## 批量测试

对 `PID / MPPI / MARL` 三种方法在 `road0..5` 上批量测试：

```bash
python airsim_occt_batch_eval.py \
  --methods pid mppi marl \
  --roads 0 1 2 3 4 5 \
  --map-dir /home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path \
  --vehicles vehicle0 vehicle1 vehicle2 vehicle3 vehicle4 \
  --step-count 200 \
  --output-dir /home/yons/Graduation/ue_bridge_occt_compact/airsim_occt_tracking_outputs
```

如果希望某条路失败后继续后面的组合，追加：

```bash
--continue-on-error
```

## 日志绘图

统一使用一个脚本绘制 actor 和铰接状态图：

```bash
python airsim_occt_plot_actor_log.py \
  --log-file /path/to/tracking_log.json \
  --modes actor hinge
```

只画 actor：

```bash
python airsim_occt_plot_actor_log.py \
  --log-file /path/to/tracking_log.json \
  --modes actor
```

只画铰接状态：

```bash
python airsim_occt_plot_actor_log.py \
  --log-file /path/to/tracking_log.json \
  --modes hinge
```

输出图保存在对应 tracking 目录下的 `plots/` 子目录。

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
