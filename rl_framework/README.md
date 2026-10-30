# RLFramework

本次课程作业使用的 PyTorch / TensorDict 强化学习组件库，提供环境与策略接口、独立智能体轨迹、同步和稀疏采样、并行整局采样、回报处理、循环序列、学习器及存储组件。

完整说明统一见 [框架架构与使用指南](docs/README.md)，包括代码组织、组件职责、数据和动作契约、同步整局 PPO 示例、场景支持范围与扩展方法。

在[作业包根目录](../README.md)安装训练依赖与框架：

```bash
python -m pip install -r requirements.txt
python -m pip install -e ./rl_framework
```

建议使用 Python 3.10。三个游戏的训练、评估和 Bot 导出命令见作业包使用说明；框架的安装信息见 `pyproject.toml`，组件行为示例见 `tests/`。
