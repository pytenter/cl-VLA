# CL-LoRA on OpenVLA

本仓库的 `openvla` 分支保存 OpenVLA-OFT 的持续学习实现：标准 LoRA、CL-LoRA 任务槽、原型回放、等预算均匀回放，以及 LIBERO / RoboTwin 数据构建和评估代码。π0.5 在独立的 [pi0.5 分支](https://github.com/pytenter/cl-VLA/tree/pi0.5) 维护；两条模型分支并行开发。

**当前是待补齐上游依赖的源码快照，不能开箱训练或评估。** 原始包的两份 `prismatic/` 都缺少整个 `models/` 目录，但训练、数据加载和推理依赖其中的 `action_heads`、`projectors`、`film_vit_wrapper`、`backbones` 等模块。原包未提供对应的上游 commit SHA；应先确认匹配原始实验的 OpenVLA-OFT 版本，再补齐并记录来源。本分支没有擅自用最新上游覆盖这些文件。

本次整理保留了现有训练、CL-LoRA、回放、数据构建和评估代码；没有重新训练模型或复现成功率。历史 README、实验记录及 `90成功率代码` 中的数字属于原始材料，不是本次发布验证的结果。

## 目录

| 路径 | 内容 |
| --- | --- |
| `openvla-oft/vla-scripts/` | `finetune.py` 标准 LoRA；`train_cl_lora.py` / `cl_lora.py` 持续学习；`replay_dataset.py` 回放加载；原型/均匀回放构建、任务槽处理与分析脚本 |
| `openvla-oft/prismatic/` | 模型接口、RLDS 管线、归一化和数据集注册；**缺少 `models/`** |
| `openvla-oft/experiments/robot/` | 模型加载、LIBERO / ALOHA 评估及数据处理 |
| `RoboTwin-main/policy/openvla-oft/` | RoboTwin 策略、任务槽路由、数据转换、TFDS builder 和评估脚本；含与主实现不同的数据配置和推理代码 |
| `RoboTwin-main/script/` | 本项目修改的评估入口和多进程结果合并脚本 |
| `openvla-oft/LIBERO/` | 带原许可证的 LIBERO 源码与 BDDL 任务定义；不含完整仿真资产和初始状态 |
| `legacy_runs/` | 原始实验启动脚本；服务器路径已替换为 `/path/to/...`，需先修改路径和参数 |
| `archive/openvla/` | 早期 `cllora代码` 与单任务 `90成功率代码`，供对照，不作为当前入口 |
| `docs/original/` | 原始中英文方法说明、结果记录和图示 |
| `docs/SOURCE_SNAPSHOT.json` | 本地源快照与整理范围记录 |

模型权重、训练数据、回放缓存、仿真资产、运行日志和 π0.5 模型实现不随此分支发布。

## 环境与运行准备

```bash
git clone --branch openvla --single-branch https://github.com/pytenter/cl-VLA.git
cd cl-VLA
cp .env.example .env
# 编辑 .env 中的模型、数据、环境和外部 benchmark 路径
```

可先运行 `python scripts/check_source_completeness.py` 查看缺失清单（当前预期退出码为2，不加载模型）。

1. 先确认上游版本，补齐两套源码所需的 `prismatic/models/`，并将来源和版本写入快照记录。两套 `prismatic` 的数据处理不同，不应直接互相覆盖。
2. 按 [OpenVLA-OFT SETUP](openvla-oft/SETUP.md)、[LIBERO 说明](openvla-oft/LIBERO.md) 和 [ALOHA 说明](openvla-oft/ALOHA.md) 准备依赖、模型权重与数据。依赖包含特定的 Transformers / dlimp fork，环境需与原始实现匹配。
3. LIBERO 评估需完整外部 LIBERO 安装、仿真资产和初始状态，并配置其路径。随附精简源码不足以创建完整仿真环境。
4. RoboTwin 评估需完整外部 RoboTwin 框架、任务配置和资产；按 [RoboTwin 策略说明](RoboTwin-main/policy/openvla-oft/openvla-oft.md) 接入本分支策略及修改过的 `script/` 文件。本分支只包含集成所需的部分 RoboTwin 文件。

每个终端只选择一个运行配置，避免两套同名 `prismatic` / `experiments` 包混入同一 Python 进程：

```bash
source .env
source scripts/env.sh libero
# 或在新的终端中重新加载 .env，再使用：
# source .env
# source scripts/env.sh robotwin
```

`scripts/env.sh` 配置源码与资源路径；依赖安装和模型补齐仍需先完成。确认 Python 实际导入的包路径后，再使用对应入口。历史命令和实验参数见 [原始 README](docs/original/README.zh-CN.md) 与 [实验记录](docs/original/REPLAY_BUG_NOTES.md)，`legacy_runs/` 里的脚本需要逐项调整后运行。

OpenVLA 的动作维度、状态字段、动作分块、归一化统计与 π0.5 不同。迁移数据或比较实验前，应核对对应 profile 的 `prismatic/vla/constants.py`、RLDS 配置、数据转换及 checkpoint 统计；不要直接复用另一模型的动作/state 规格。

## 协作与来源

提交从 `openvla` 创建功能分支，通过 Pull Request 合入 `openvla`；完整流程见 [协作说明](docs/COLLABORATION.md)。

本项目基于 OpenVLA / OpenVLA-OFT，使用 LIBERO 和 RoboTwin。已保留 [OpenVLA-OFT MIT 许可证](openvla-oft/LICENSE)、[RoboTwin MIT 许可证](RoboTwin-main/LICENSE)、[LIBERO MIT 许可证](openvla-oft/LIBERO/LICENSE) 和 [RLDS builder MIT 许可证](RoboTwin-main/policy/openvla-oft/rlds_dataset_builder/LICENSE)。引用与历史方法说明见 [原始文档](docs/original/README.md)；模型权重及外部数据请分别遵循其来源要求。
