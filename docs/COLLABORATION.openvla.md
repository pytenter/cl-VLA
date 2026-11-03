> 保留的模型分支说明。main 的统一入口与协作规则以 [COLLABORATION.md](COLLABORATION.md) 为准。

# OpenVLA 分支协作

`openvla` 维护 OpenVLA-OFT / CL-LoRA；`pi0.5` 维护 π0.5。两条分支并行开发，不将整个模型分支互相合并。共享实验协议或数据接口的改动先在 PR 中说明，再按需要分别提交到各模型分支。

## 获取代码与提交

```bash
git clone --branch openvla --single-branch https://github.com/pytenter/cl-VLA.git
cd cl-VLA
git switch -c feature/openvla-your-change

# 完成修改并检查差异后提交
git add <本次修改的文件>
git commit -m "Describe the OpenVLA change"
git push -u origin feature/openvla-your-change
```

创建 Pull Request 时将 **base 设为 `openvla`**。说明解决的问题、涉及 LIBERO 还是 RoboTwin、验证命令与结果，以及尚未验证的部分。不要把整理源码、通过静态检查或已有历史日志写成新训练/评估结果。

## 开发约定

- 每个新终端先 `source .env`；LIBERO 与 RoboTwin 分别使用 `source scripts/env.sh libero` 和 `source scripts/env.sh robotwin`；切换时开新终端，确认 Python 导入的 `prismatic` 和 `experiments` 来自预期目录。
- 两套 OpenVLA 源码有不同的数据处理与推理实现。修改公共逻辑时审查另一份是否需要同步，并在 PR 交代；不要仅因文件同名就覆盖。
- 本地资源路径写入 `.env`；提交 `.env.example` 中的变量说明，不提交 `.env`、账号密钥、模型、数据、缓存或运行产物。`legacy_runs/` 是需要修改路径的历史脚本。
- 修改数据接口时写清图像键、动作/state 维度、单位、绝对/相对动作、分块长度和归一化来源。OpenVLA 与 π0.5 的这些规格需分别核对。
- 新实验记录所用模型、代码版本、任务顺序、数据、checkpoint、评估任务槽、episode 数与随机种子，便于比较和复查。

## 当前需要补齐的依赖

原始包的两套 `prismatic/models/` 都缺失，上游 commit SHA 也未提供。补齐前先确认与原实验匹配的源码版本；补齐后保留许可证并更新 `SOURCE_SNAPSHOT.json` 的来源记录。不要默认拉取上游 HEAD 后宣称与原实验一致。

LIBERO 缺少完整仿真资产和初始状态；RoboTwin 只附策略和部分评估脚本，需要外部完整安装。按改动范围完成可用的语法、导入或小规模运行检查，并明确记录依赖缺失导致无法运行的检查。GPU 训练和正式成功率评估应单独记录，历史 `90成功率代码` 目录名不代表本分支已复现该结果。
