# π0.5 核心源码

本目录保留实际实验所用的 OpenPI / RoboTwin π0.5 核心、客户端、训练器、依赖锁文件和许可证。

- [项目与分支说明](../README.md)
- [LIBERO 与 Kinova 复现入口](../docs/REPRODUCTION.md)
- [Apache-2.0 许可证](LICENSE)

模型实现在 `src/openpi/models/`。π0 与 π0.5 共享部分实现文件，因此文件名出现 `pi0.py` 不表示本分支实验使用了π0：本项目实际实验配置显式设置 `pi05=True`。

当前机器路径通过 `openpi.shared.project_paths` 和仓库的 `.env.example` 配置。原上游 RoboTwin task 配置及示例脚本保留供参考；已复现的 LIBERO / Kinova 实验使用仓库 `experiments/` 中的适配代码。
