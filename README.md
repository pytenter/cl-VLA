# CL-VLA：π0.5 与 OpenVLA 的 CL-LoRA 实验

`main` 集成 π0.5 / OpenPI 与 OpenVLA-OFT 两套实现，保留普通 LoRA、CL-LoRA、任务参数 bank 和回放相关代码。`pi0.5` 与 `openvla` 模型分支继续保留，协作流程见 [分支协作](docs/COLLABORATION.md)。两套模型使用独立 Python 环境、配置和数据适配。

| 模型 | 当前内容与验证范围 |
| --- | --- |
| π0.5 / JAX | 已完成 LIBERO 四任务五组连续学习实验，以及 Kinova Gen3 普通 LoRA 有效 2000 步离线微调；保留实际实验代码与精简结果 |
| OpenVLA-OFT / PyTorch | 保留 LIBERO / RoboTwin 的训练、CL-LoRA、回放、数据构建和评估代码；两套 `prismatic/models/` 均缺失，当前是待补齐依赖的源码快照，不能开箱训练或评估 |

本次整合没有重新训练或评估模型。仓库不包含训练轨迹、基础模型权重、训练检查点、回放样本、完整仿真资产、缓存或个人配置。

## 目录

| 路径 | 内容 |
| --- | --- |
| `pi05/` | π0.5 / OpenPI 核心、普通与 CL 训练器、`openpi-client` |
| `experiments/libero_fourway/` | π0.5 普通 LoRA、CL-LoRA 无回放、原型回放、等预算均匀回放 |
| `experiments/libero_plain_prototype/` | π0.5 第五组：普通 LoRA + 固定原型回放 |
| `experiments/kinova/`、`data/kinova/` | π0.5 腕部单视角微调、离线评估、数据转换及归一化资产 |
| `results/` | π0.5 LIBERO 阶段结果与 Kinova 最终结果摘要 |
| `openvla-oft/` | OpenVLA-OFT 主实现；`vla-scripts/` 含训练和回放，`experiments/robot/` 含评估与数据处理 |
| `RoboTwin-main/policy/openvla-oft/`、`RoboTwin-main/script/` | OpenVLA 的 RoboTwin 策略、任务槽、数据构建和部分评估集成 |
| `openvla-oft/LIBERO/` | 精简 LIBERO 源码与 BDDL 定义；不含完整资产和初始状态 |
| `legacy_runs/`、`archive/openvla/` | OpenVLA 历史启动脚本与早期实现，供对照；运行前需审查路径和参数 |
| `docs/original/` | OpenVLA 原始方法说明、实验记录和图示 |
| `scripts/env.sh` | 统一环境入口，按模型与实验选择 profile |

## 获取代码与环境选择

```bash
git clone --branch main https://github.com/pytenter/cl-VLA.git
cd cl-VLA
```

分别建立 π0.5 的 JAX 环境与 OpenVLA 的 PyTorch 环境，按各自锁定依赖安装；不要把两套依赖合并到一个环境。每个终端先激活所需 Python 环境，再从仓库根目录加载对应本地配置与 profile。环境脚本只配置路径和运行变量，不安装依赖、下载资产或补齐模型源码。切换 profile 应新开终端；统一入口会拒绝同一终端切换 profile，以免 Python 包路径混用。

### π0.5：LIBERO 与 Kinova

依赖以 `pi05/pyproject.toml`、`pi05/uv.lock` 与既有运行记录为准。既有关键版本为 Python 3.11、JAX 0.5.0、Flax 0.10.2、Orbax Checkpoint 0.11.1、NumPy 1.26.4；LeRobot 固定到 `a445d9c9da6bea99a8972daa4fe1fdd053d711d2`。LIBERO 仿真还需对应依赖与固定资产。

```bash
# 在已激活的独立 π0.5 环境中，从仓库根目录执行
cp .env.pi05.example .env.pi05
# 编辑 .env.pi05 中的模型、数据和运行路径
source .env.pi05
source scripts/env.sh libero-fourway
```

π0.5 可选 profile 为 `libero-fourway`、`libero-plain-prototype`、`kinova`，每个 profile 使用独立终端。先阅读 [π0.5 复现说明](docs/REPRODUCTION.md)，再使用对应训练或评估入口。历史 supervisor 仍依赖前序报告、运行目录布局和续训状态，尚未验证全新机器上的完整冷启动流程。

### OpenVLA：LIBERO 与 RoboTwin

**先补齐源码依赖，再尝试训练或评估。** 原包的 `openvla-oft/prismatic/` 和 `RoboTwin-main/policy/openvla-oft/prismatic/` 都缺少整个 `models/` 目录，训练、数据加载与推理所需的 `action_heads`、`projectors`、`film_vit_wrapper`、`backbones` 等模块因此不可用。原包也未提供对应上游 commit SHA；需先确认与原实验匹配的 OpenVLA-OFT 版本，保留许可并更新 [源码来源记录](docs/SOURCE_SNAPSHOT.json)。两份实现的数据处理与推理存在差异，不应互相覆盖或直接用上游最新版本替代。

```bash
# 在已激活的独立 OpenVLA 环境中，从仓库根目录执行
cp .env.openvla.example .env.openvla
# 编辑 .env.openvla 中的模型、数据和外部 benchmark 路径
source .env.openvla
source scripts/env.sh openvla-libero

# 无需加载模型的源码检查；当前缺失状态的预期退出码为 2
python scripts/check_source_completeness.py
```

RoboTwin 使用新的终端和同一模型的本地配置，选择 `source scripts/env.sh openvla-robotwin`。旧 `libero` / `robotwin` 名称仅作为 OpenVLA profile 的兼容别名。每个进程只加载对应的一套 `prismatic` / `experiments`，运行前确认实际导入路径。

按 [OpenVLA-OFT SETUP](openvla-oft/SETUP.md)、[LIBERO 说明](openvla-oft/LIBERO.md) 和 [ALOHA 说明](openvla-oft/ALOHA.md) 准备匹配的依赖、权重和数据，注意特定 Transformers / dlimp fork。LIBERO 评估还需外部完整安装、仿真资产与初始状态；RoboTwin 需外部完整框架、任务配置和资产，并按 [策略说明](RoboTwin-main/policy/openvla-oft/openvla-oft.md) 接入本仓库策略及修改过的脚本。

## 现有实验结果

π0.5 LIBERO 按 A → B → C → D 训练：`libero_spatial/6`、`libero_spatial/0`、`libero_object/8`、`libero_goal/7`。五组均为训练种子 42，每阶段 10,000 步、batch 2；每阶段评估所有已学任务，每任务 50 次。第五组复用 CL-LoRA 原型组已选定的记忆，没有用普通 LoRA 重新选择原型。最终 D 阶段四任务平均成功率中，CL-LoRA 原型组为 **91%**，普通 LoRA 固定原型组为 **90%**。完整矩阵见 [LIBERO 结果](results/libero/)；单种子结果不支持跨种子的显著性结论。

π0.5 Kinova 采用普通 LoRA、腕部图像、真实 state28/action7、horizon50，微批次 1、累积 4 次。有效轨迹为原第 500 步完整状态续训 1500 次更新，最佳检查点为第 2000 步；固定验证计划的选择指标由基线 **0.33544426** 降至 **0.02084956**。这是离线动作预测误差，不能解读为真机任务成功率。详情见 [Kinova 结果](results/kinova/FINAL_RESULT_zh.md)。

OpenVLA 的历史 README、实验记录及 `90成功率代码` 中的数字属于原始材料，本次整合未重新验证其训练或成功率。历史方法和参数见 [原始 README](docs/original/README.zh-CN.md) 与 [回放实验记录](docs/original/REPLAY_BUG_NOTES.md)。

## 协作、协议与来源

`main` 是两套模型的集成分支，`pi0.5` 与 `openvla` 保留模型开发线。功能分支根据实际基线和交付范围，向 `main` 或对应模型分支提交 PR；模型分支上的改动经检查后可继续集成到 `main`。详见 [协作说明](docs/COLLABORATION.md)。

跨模型共享数据时，逐项核对图像键、相机语义、state/action 顺序与单位、动作参考系、分块长度、归一化来源、数据划分和评估规则。π0.5 LIBERO 的两视角/state8/action7/horizon10、Kinova 的腕部视角/state28/action7/horizon50 与 OpenVLA 各 profile 的协议需要分别确认。

原模型分支整理说明保留为 [π0.5 原说明](README.pi0.5.md) 和 [OpenVLA 原说明](README.openvla.md)，用于追溯；当前 `main` 的分支协作和环境入口以本文及协作说明为准。模型专用脚本保留为 `scripts/env_pi05.sh` 与 `scripts/env_openvla.sh`，日常使用统一入口。

第三方许可按各目录保留，来源概览见 [第三方声明](THIRD_PARTY_NOTICES.md)，包括 [OpenVLA-OFT MIT](openvla-oft/LICENSE)、[RoboTwin MIT](RoboTwin-main/LICENSE)、[LIBERO MIT](openvla-oft/LIBERO/LICENSE) 和 [RLDS builder MIT](RoboTwin-main/policy/openvla-oft/rlds_dataset_builder/LICENSE)。权重及外部数据分别遵循其来源要求。
