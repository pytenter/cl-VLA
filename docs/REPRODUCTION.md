# 复现范围与入口

本仓库提供已运行实验的源码与精简结果，未打包数据、权重和历史运行状态。发布时未重新执行完整训练，也未验证全新机器从零安装到最终结果的一键流程。以下命令用于说明真实 CLI 和执行顺序，模型执行须先满足对应前置条件。

## 运行依赖与路径

核心位于 `pi05/`：模型实现在 `src/openpi/`，普通与 CL 训练器分别为 `scripts/train.py`、`scripts/train_cl_lora.py`，客户端在 `packages/openpi-client/`。LIBERO 和 Kinova 实验均在核心外增加数据/模型适配，不能仅运行核心默认 RoboTwin 配置替代现有实验。

依赖以 `pi05/pyproject.toml`、`pi05/uv.lock` 及发布的运行依赖记录为准。既有环境的关键版本为 Python 3.11、JAX 0.5.0、Flax 0.10.2、Orbax Checkpoint 0.11.1、NumPy 1.26.4；LeRobot 固定到 `a445d9c9da6bea99a8972daa4fe1fdd053d711d2`。LIBERO 仿真还需要对应的 MuJoCo、robosuite、LIBERO 包与固定资产。

按 `.env.pi05.example` 与 `scripts/env.sh` 配置当前机器路径，使核心、客户端和所选实验模块可导入。实验运行目录与仓库源码目录应区分：历史 supervisor 会寻找运行根目录下的 `scripts/`、`reports/`、`configs/`、`assets/` 等布局。环境变量模板不代表这些运行输入已经生成。

基础参数使用官方 `pi05_base`。准备独立的模型、数据和运行目录，保持原始输入只读。具体入口和可用参数可先查看：

```bash
python experiments/libero_fourway/train_stage.py --help
python experiments/libero_fourway/build_memory.py --help
python experiments/libero_fourway/serve_libero.py --help
python experiments/libero_fourway/evaluate_libero.py --help
python experiments/libero_plain_prototype/train_stage.py --help
python experiments/kinova/train_kinova.py --help
python experiments/kinova/evaluate_offline.py --help
```

Kinova 模块在解析参数前导入部分运行依赖，因此查询其帮助也应在依赖已安装的环境执行。

## LIBERO：四任务与五组

| 阶段 | suite / task ID | 任务 |
| --- | --- | --- |
| A | `libero_spatial / 6` | 黑碗：cookie box 旁 → plate |
| B | `libero_spatial / 0` | 黑碗：plate 与 ramekin 之间 → plate |
| C | `libero_object / 8` | chocolate pudding → basket |
| D | `libero_goal / 7` | turn on the stove |

四组入口支持 `plain`、`noreplay`、`prototype`、`uniform`；第五组入口支持 `plain_prototype`。共同协议为 state8、真实 action7/内部 padding32、两路图像、horizon10、离散 state tokens、seed42、每阶段 10,000 步、batch2、无 EMA/KD。LIBERO 动作已经是 delta，不能再做一次绝对到相对动作转换。

数据与训练按下列顺序准备：

1. 使用 `physical-intelligence/libero` 固定 revision `a4336d589d589045d1c56423ffdf3b88a0e19b1f`，先取得 `info.json`、`tasks.jsonl`、`episodes.jsonl` 元数据，放入运行根的 `data/physical-intelligence-libero/`。`fetch_data.py` 依赖这些元数据，随后下载四任务 parquet 并生成 `reports/source_download_manifest.json`。
2. `prepare_tasks.py` 读取上述 manifest，生成 `datasets/libero_fourway/task_{A,B,C,D}`、episode 来源映射及每任务 train-only 统计。原流程共 191 条 episode、22,680 帧；已有目标输出时不要重复创建。
3. 运行 `validate_data.py` 与 `preflight.py`，核对数据、动作、相机、配置和路径；这些检查也有既定目录/报告前置条件。
4. 每组依次训练 A/B/C/D。A 从基础参数开始，B/C/D 从本组前一阶段参数开始，各阶段重建优化器。CL 入口会安装 `stable_training.py` 的初始化与 task bank 更新保护；应保留该调用。
5. 每阶段评估所有已学任务。`prototype` 和 `uniform` 在 A/B/C 后构建供后续阶段使用的记忆；均匀组通过 `--reference-buffer` 匹配原型组相同 episode 和每 episode 样本数。不得引入未来任务记忆。

在数据、统计、基础权重、模块路径和运行目录均就绪后，单阶段训练接口例如：

```bash
python experiments/libero_fourway/train_stage.py \
  --group prototype --stage A --steps 10000 --batch-size 2 \
  --weights "$PI05_BASE_PARAMS"
```

`PI05_BASE_PARAMS` 在此示例中指向基础模型的 `params/` 目录，须由当前环境设置。输出由该实验的 `PI05_LIBERO_ROOT` 与配置决定；此示例不是完整的数据准备或冷启动脚本。

评估分成两个进程：`serve_libero.py` 加载 checkpoint 和配置工厂 `experiment_config:config_from_name`；`evaluate_libero.py` 在仿真环境连接它。服务端 `--checkpoint` 接收包含 `params/` 的步目录，记忆构建器则接受步目录或 `params/`。评估任务必须使用自身统计和正确 bank ID（A/B/C/D 对应 0/1/2/3），并校验语言指令。固定评估为每任务 50 次、eval seed7、policy seed0、warmup10、replan5、10 次去噪。

第五组复用原四组的任务数据、统计和 `prototype/task_{A,B,C}` 记忆：分别 219、225、261 个样本。它从基础参数重训自己的 A 阶段，不从旧 plain 或 CL checkpoint 初始化；普通 LoRA 参数参与训练，B/C/D 通过 `plain_replay_training.py` 加入回放。删除旧记忆或自行重选后，不能称为原第五组的同一对照。

五组共 20 个正式训练阶段、50 场评估、2500 次 rollout。结果中的最终均值是 D 阶段四任务均值；阶段末 AUC/FWT/NBT 的具体定义见 `results/libero/RESULTS_stage_end.json`，不能与其他论文同名指标直接等同。所有实验只有一个训练种子。

## Kinova：数据、训练与验证

`data/kinova/` 保留 RLDS → LeRobot 转换与适配实现。当前数据划分为 train 60 条/8011 帧、validation 20 条/2754 帧、test 27 条/3758 帧。转换保留 episode 来源、边界和原始时间信息，归一化只用训练集；原始轨迹需另外取得。

| 输入/标签 | 当前协议 |
| --- | --- |
| 图像 | 只启用腕部；另外两路占位且 image mask 关闭 |
| 状态 | 7 关节位置 + 7 速度 + 7 扭矩 + tool xyz/Euler + gripper，共 28 维 |
| 动作 | `dx, dy, dz`（m）、三个角增量（rad）、夹爪开度，共 7 维 |
| 模型 | state/action padding32，horizon50，离散状态 tokens |
| 损失 | 仅真实动作维度与有效时间位置，排除 episode 尾部 padding |

`kinova_config.py` / `kinova_model.py` 是本次训练适配，不能用早期候选 `kinova_adapter.py` 的默认双视角配置替换。配置为普通 LoRA rank16/alpha32/dropout0，训练 LoRA 与动作头；微批次1、累积4、seed42，LoRA/head 峰值学习率分别为 1e-4/1e-5，warmup100，AdamW、clip1、无 EMA、无 CL bank 或回放。

`train_kinova.py` 提供 `--mode smoke|pilot`、`--resume`、`--stop-at 1000|1500|2000` 和 `--verify-full-state`；`evaluate_offline.py` 提供 `self-test`、`run`、`select`。正式模型执行会检查资源策略；评估的 `--resource-approved` 应在资源检查通过后使用，不应用来绕过检查。

实际有效 2000 步由先前完整第 500 步状态续训 1500 步而成，保留 Adam 和学习率计数；早期失败分支、smoke 和保存探针不重复计入。每 500 步保存并评估，最终选择第 2000 步。验证固定为 20 条轨迹、每检查点 60 个样本，使用相同输入、噪声与训练集统计；测试集未用于选择。

选择指标为前 10 个有效动作的归一化误差，XYZ、Euler、夹爪三组等权，再按既定样本/episode 规则聚合，越低越好。基线 0.33544426，2000 步 0.02084956；这是离线预测改善。当前数据同步、控制器动作接口/参考系仍待确认，结果不构成闭环真机成功率验证。

## 历史工作流的依赖

`run_experiment.py` / `run_workflow.py` 保存了原执行顺序与恢复逻辑，但也引用了固定运行布局。Kinova 的 `prepare_training.py`、`recover_smoke.py` 和 supervisor 还会读取前序 probe、CPU 检查报告、同图基线及第 500 步完整 checkpoint。仓库中的精简结果不具备这些运行状态，直接启动 supervisor 可能因缺少证据或输入而退出。

迁移时先列出并准备依赖，再适配路径和运行布局；需要从基础模型开始的新轨迹时，应另建运行目录并审查初始化/验证入口。路径迁移不能改写模型数学、数据划分或历史结果；新的冷启动执行通过后再记录其命令与验证范围。
