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
│       ├── default.yaml           # 默认配置
│       ├── actor_baseline.yaml    # Actor模型基线
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

### AirSim接口 (`airsim_occt_airsim_io.py`)
处理与AirSim的通信，包括车辆状态读取、控制命令发送、坐标转换和数据同步。

### 道路投影器 (`airsim_occt_map_projector.py`)
将车辆位置投影到道路中心线，支持Frenet坐标系转换（s, y坐标）和道路曲率计算。

## 快速开始

### 安装依赖

```bash
pip install airsim torch numpy pyyaml matplotlib msgpack-rpc
```

### 基本使用

```python
from airsim_occt_env_demo import create_env_and_run

# 运行演示程序
create_env_and_run(
    config_path='configs/algorithm/default.yaml',
    controller_type='constant',  # 或 'actor'
    actor_model_path=None,       # Actor模型路径
    num_steps=1000
)
```

### 配置文件示例

```yaml
# configs/algorithm/default.yaml
env:
  map_path: "path/to/map.xml"
  n_vehicles: 4
  dt: 0.1

vehicles:
  - name: "leader"
    init_position: [0, 0, 0]
    init_speed: 10.0
  - name: "follower1"
    init_position: [-5, 0, 0]
    init_speed: 10.0

control:
  longitudinal:
    kp: 0.5
    ki: 0.1
    kd: 0.0
  lateral:
    k_gain: 2.0
```

## 使用示例

### 演示程序

```bash
# 运行恒定速度控制
python airsim_occt_env_demo.py --config configs/algorithm/constant_baseline.yaml

# 运行Actor模型控制
python airsim_occt_env_demo.py --config configs/algorithm/actor_baseline.yaml --model path/to/actor.pth
```

### 可视化

```bash
# 可视化运行日志
python airsim_occt_plot_actor_log.py --log_path airsim_occt_tracking_outputs/
```

## 训练和部署

### MAPPO/IPPO训练

```bash
cd ivs_python_example
python mappo_ippo_occt.py --config config/mappo_occt_3_followers
```

### 训练特性

- 支持MAPPO（共享参数）和IPPO（独立参数）
- 基于TorchRL框架实现
- 使用VMAS仿真器进行高效并行训练
- 支持中心化和去中心化Critic网络

### 观察空间

- **自身状态**：位置、速度、加速度
- **邻居车辆**：相对距离、相对速度
- **道路信息**：短期路径点、曲率
- **历史信息**：过去N步的状态

### 奖励函数

- 路径跟踪奖励（横向误差）
- 速度跟踪奖励
- 车间距保持奖励
- 碰撞惩罚
- 编队协同奖励

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
