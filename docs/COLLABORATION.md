# 分支协作

`main` 是 CL-VLA 的集成分支，同时包含 π0.5 / OpenPI 与 OpenVLA-OFT；`pi0.5` 和 `openvla` 继续保留各模型的开发线。两套源码可通过明确的合并和 PR 集成到 `main`，运行环境与数据适配仍按模型分别维护。

## 选择基线与 PR 目标

默认从最新 `main` 创建功能分支，适合仓库公共入口、协作文档、共享协议与直接交付集成版本的改动：

```bash
git clone --branch main https://github.com/pytenter/cl-VLA.git
cd cl-VLA
git switch -c feature/your-change
```

完成修改和适当验证后，推送功能分支并创建 **base = `main`** 的 PR。已有本地仓库时，先获取远端更新并确认实际基线。

若改动需要先在模型开发线维护或验证，则从 `pi0.5` / `openvla` 创建功能分支，PR 的 base 选对应模型分支；通过检查后再将所需改动集成到 `main`。PR 目标由实际提交和交付范围决定，避免把无关模型提交带入单模型 PR。公共协议变更须写明两套模型的影响和适配范围。

PR 应说明具体问题、改变后的行为、验证命令与结果、尚未验证的部分。涉及模型结构、冻结边界、任务 bank、回放、归一化、动作定义、数据划分或评估规则时，列出受影响实验并明确旧结果是否适用。源码整理、CPU 静态检查、导入检查、小型运行和正式 GPU 训练/评估结果应分别标注。

## 环境与源码边界

- π0.5 使用独立 JAX 环境、`.env.pi05.example` 和 `libero-fourway` / `libero-plain-prototype` / `kinova` profile；OpenVLA 使用独立 PyTorch 环境、`.env.openvla.example` 和 `openvla-libero` / `openvla-robotwin` profile。不要混装两套依赖。
- 在仓库根目录复制所需模板为本地 `.env.pi05` 或 `.env.openvla`，编辑后先激活相应 Python 环境，再 source 本地配置与 `scripts/env.sh <profile>`。每个终端只使用一个 profile；切换时新开终端。旧 `libero` / `robotwin` 仅作为 OpenVLA 兼容别名。
- 合并后 π0.5 仍位于 `pi05/`、`experiments/`、`data/kinova/` 和 `results/`；OpenVLA 位于 `openvla-oft/`、`RoboTwin-main/`、`archive/openvla/` 和 `legacy_runs/`。审查同名 Python 包和脚本的实际来源，不以同名为由覆盖文件。
- OpenVLA 的两套 `prismatic` 含不同数据处理与推理实现。修改公共逻辑时检查另一份是否需要同步，并在 PR 说明。

当前两套 OpenVLA `prismatic/models/` 均缺失，原始包未提供对应上游 commit SHA。补齐前先确认与原实验匹配的版本，保留许可证并更新 `docs/SOURCE_SNAPSHOT.json`；补齐和静态检查通过不等于复现历史结果。LIBERO 缺完整仿真资产和初始状态；RoboTwin 只附策略和部分评估脚本，需外部完整安装。按改动范围完成可行检查，记录缺失依赖阻止了哪些验证。

历史 `README.pi0.5.md` / `README.openvla.md` 以及 `docs/COLLABORATION.pi0.5.md` / `docs/COLLABORATION.openvla.md` 保留整理前说明。当前集成分支的入口和协作规则以根 README 与本文为准。

## 共享数据与评估协议

跨模型共享的是明确的数据协议与评估约定：图像字段、相机语义、state/action 顺序与单位、绝对/相对动作、参考系、episode 边界、分块长度、数据划分、训练集统计和固定评估样本。接口变更在 PR 中给出前后 schema、兼容方式和每套模型的适配范围。

π0.5 LIBERO 的两视角/state8/action7/horizon10 与当前 Kinova 的腕部视角/state28/action7/horizon50 是不同协议。OpenVLA 需按 profile 核对 `prismatic/vla/constants.py`、RLDS 配置、转换代码与 checkpoint 统计；不能直接复用另一模型的规格。LIBERO 第五组复用既有 CL-LoRA 原型记忆，重新选择记忆应另立实验记录。

## 文件与实验记录

数据、权重、checkpoint、回放 NPZ、视频、缓存、运行产物、个人配置和凭据不进入 Git。数据与模型通过独立存储共享，仓库保留来源、版本和校验信息。本地路径写入 `.env.pi05` / `.env.openvla`；提交模板中的变量说明。提交前检查 `git diff --cached`，确认没有个人路径、凭据或大文件。`legacy_runs/` 是需审查路径和参数的历史脚本。

保留既有实验结果，新实验使用独立运行目录。记录模型、代码提交、配置、数据/权重版本、任务顺序、种子、checkpoint、评估任务槽、episode 数与实际验证范围。不要覆盖已有运行目录，也不要把失败探针的更新次数计入有效训练轨迹。

π0.5 已有 LIBERO 五组与 Kinova 有效 2000 步结果，应保留原始评估定义；Kinova 离线误差不能写成真机成功率。OpenVLA 的历史目录名、README 数字或旧日志不能写成本次整合新复现的训练或成功率结果。
