# 来源与已有许可证

- `pi05/` 来自项目原始压缩包中的 `RoboTwin-main/policy/pi05`，包含 OpenPI 模型和客户端。其 Apache-2.0 许可证保留在 `pi05/LICENSE`。
- RoboTwin 的 MIT 许可证副本保留在 `licenses/RoboTwin-MIT.txt`。
- LeRobot 作为依赖引用，现有实验固定到提交 `a445d9c9da6bea99a8972daa4fe1fdd053d711d2`，源码未重复 vendoring。
- `docs/runtime_source_sha256.json` 记录从已完成实验环境读取的源码哈希。该快照包含已验证的相邻区间测试修复；核心训练与模型文件未额外改写算法。
- `docs/publication_path_changes.json` 记录发布副本中环境路径适配的文件。适配前后对路径替换以外的 AST 作过一致性检查。发布不代表重新执行了一轮模型训练。
- 数据、模型权重、仿真资产、SSH配置和旧运行目录未包含在本仓库中。
