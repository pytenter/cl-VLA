# π0.5：CL-LoRA / LIBERO 与 Kinova 离线微调

本分支 `pi0.5` 保存现有 π0.5 工程：LIBERO 四任务连续学习的五组实验，以及 Kinova Gen3 单任务普通 LoRA 的有效 2000 步微调代码和结果摘要。核心采用 OpenPI 的 JAX 实现，并保留 CL-LoRA、任务参数 bank 和回放实现。

队友的 OpenVLA 工作在其独立分支协作；本分支不包含 SmolVLA 或 OpenVLA 实现。

## 内容

| 目录 | 用途 |
| --- | --- |
| `pi05/` | 实际运行使用的 π0.5 / OpenPI 核心、训练器和 `openpi-client` |
| `experiments/libero_fourway/` | 普通 LoRA、CL-LoRA 无回放、CL-LoRA 原型回放、CL-LoRA 均匀回放 |
| `experiments/libero_plain_prototype/` | 第五组：普通 LoRA + 固定原型回放 |
| `experiments/kinova/` | 腕部单视角训练、离线评估、分段续训与检查点验证 |
| `data/kinova/` | 数据转换/适配代码与小型归一化资产 |
| `results/` | LIBERO 阶段结果及 Kinova 最终结果摘要 |
| `scripts/env.sh`、`.env.example` | 本机运行路径配置模板 |

仓库不包含训练轨迹、基础模型权重、训练检查点、回放样本、仿真大资产或缓存。源码来源与第三方许可按各目录中的许可文件保留。

## 已完成实验

LIBERO 按 A → B → C → D 训练：`libero_spatial/6`、`libero_spatial/0`、`libero_object/8`、`libero_goal/7`。五组均使用单个训练种子 42，每阶段 10,000 步、batch 2；每阶段评估所有已学任务，每任务 50 次。第五组使用 CL-LoRA 原型组已选定的同一批记忆，并未用普通 LoRA 重新选择原型。最终 D 阶段四任务平均成功率中，CL-LoRA 原型组为 91%，普通 LoRA 固定原型组为 90%。完整矩阵见 [LIBERO 结果](results/libero/)；单种子结果不支持跨种子的显著性结论。

Kinova 采用普通 LoRA，腕部图像、28 维真实状态、7 维真实动作、horizon 50，微批次 1、累积 4 次。有效轨迹为原第 500 步完整状态加上本次 1500 次更新，最佳检查点为第 2000 步：固定验证计划的选择指标由基线 **0.33544426** 降至 **0.02084956**。这是离线动作预测误差，不能解读为真机任务成功率。详情见 [Kinova 结果](results/kinova/FINAL_RESULT_zh.md)。

## 使用与协作

```bash
git clone --branch pi0.5 https://github.com/pytenter/cl-VLA.git
cd cl-VLA
```

先阅读 [复现说明](docs/REPRODUCTION.md)，再配置本机路径、固定依赖、数据与权重。当前发布保留了实际实验代码；历史 supervisor 仍有前序报告、运行布局与续训状态依赖，尚未验证全新机器上的冷启动流程。

开发从 `pi0.5` 创建功能分支，PR 的目标分支选择 `pi0.5`，具体约定见 [协作说明](docs/COLLABORATION.md)。
