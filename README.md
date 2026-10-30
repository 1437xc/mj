# 强化学习课程：三个 Botzone 游戏示例

这套示例使用给定的RL框架训练国标麻将、双升和掼蛋。学生可任选一个游戏，修改特征、网络、算法和超参数。默认训练是多进程同步PPO：一个 learner、多个 CPU actor；四个玩家使用同一份模型参数的策略，收集各自轨迹用于训练。双升、掼蛋只学习出牌，开局阶段的决策使用手写规则。

## 交付目录与环境

解压示例代码包后，在包含 `rl_framework/`、`mahjong/`、`tractor/`、`guandan/` 的根目录运行下文命令。三个游戏各自独立，代码不互相依赖。

| 目录/文件 | 用途 |
| --- | --- |
| `rl_framework/` | 通用RL框架，包括环境接口、采样器、GAE、PPO、策略接口等组件 |
| `<game>/gamecore/` | 推进规则与结算；不负责神经网络输入 |
| `<game>/features.py` | 建议修改的 NumPy 特征；训练和 Bot 共用 |
| `<game>/model.py` | 建议修改的 PyTorch 网络；训练和 Bot 共用 |
| `<game>/actions.py`、`observation.py` | 规则动作、候选或麻将私有状态 |
| `<game>/env.py` | GameCore → 框架 MultiEnv/TensorDict |
| `<game>/policy.py`、`train.py` | 策略包装、多进程PPO训练 |
| `<game>/heuristics.py` | 双升/掼蛋开局阶段的人工策略 |
| `<game>/botzone/` | 协议重放、推理和长时运行入口 |
| `<game>/export_bot.py` | 导出源码ZIP作为Bot代码、导出权重 |
| `<game>/tests/` | 游戏规则及训练/部署协议一致性测试代码 |

训练推荐独立 Python 3.10 环境，依赖统一列在 `requirements.txt`。其中 `PyMahjongGB` 不限版本，由 pip 安装可用的最新版。Bot 使用 Botzone 提供的运行环境。

```bash
python -m pip install -r requirements.txt
python -m pip install -e ./rl_framework
```

PyMahjongGB 的 Python 导入名为 `MahjongGB`。本地训练通过 requirements 安装，Bot 使用平台已安装的库。如果 pip 需要从源码构建原生扩展，请准备 C++ 编译器和 Python 开发头文件，推荐在 Linux 服务器训练。

## 从训练到提交

以下命令以 `tractor` 为例；替换成 `mahjong` 或 `guandan` 即可。

```bash
# 先完成少量完整局，确认接口与依赖
python -m tractor.train --output runs/tractor_smoke --iterations 3 --actors 2
# 约 15 分钟短训；时间在完整更新后检查，可能略超时
python -m tractor.train --output runs/tractor_15m --seconds 900 --actors 4
# 恢复 checkpoint，继续 20 次更新；输出目录必须是新目录
python -m tractor.train --resume runs/tractor_15m/checkpoint.pt --output runs/tractor_resume --iterations 20
# 固定种子，模型队伍对随机合法候选基线
python -m tractor.evaluate --checkpoint runs/tractor_15m/checkpoint.pt --output runs/eval.json --episodes 20
# 同一配置下随机初始化网络的对照
python -m tractor.evaluate --checkpoint runs/tractor_15m/checkpoint.pt --output runs/eval_initial.json --episodes 20 --untrained
# 导出源码和单独的 CPU 权重
python -m tractor.export_bot --checkpoint runs/tractor_15m/checkpoint.pt --output submissions/tractor
```

完整训练 checkpoint 保存模型、优化器、计数器、主进程 RNG、下一局种子编号。配置中的 `iterations` 是本次额外更新次数，`seconds` 是软时间预算。

`metrics.jsonl` 包含实际环境步数、agent 决策数、梯度更新数、已完成局数、actor/learner PID、参数更新检查和 loss。候选溢出计数是各 actor 生命周期累计值。整局模式会等待最慢 actor；单局超过 `actor_timeout` 时明确报错，不会把不完整局作为完整数据继续训练。每次更新后原子替换 checkpoint。

提交时：

1. 把导出的 `tractor_model.pt` 放入 Botzone 个人存储根目录，Bot 通过相对路径 `data/tractor_model.pt` 读取。
2. 上传 `tractor_bot.zip` 作为 Python 3.6 源码，ZIP 根目录有 `__main__.py`。训练 checkpoint 不是部署权重，不能代替 `*_model.pt`。
3. 双升/掼蛋选默认 JSON 交互；麻将选简单交互。代码每次响应后输出长时运行标记。首次调用支持历史重放，后续长时请求接收单条当前 request。
4. 麻将请选择平台上的 **Chinese-Standard-Mahjong** 即 136 张无花复式国标麻将，不支持花牌/144 张模式。本地用 `BOT_MODEL_PATH` 可覆盖权重位置；平台不需要此变量。

`python -m <game>` 是源码推理入口；它需要导出生成的 `bot_config.json`。实际提交Bot请直接使用导出的 ZIP。权重路径、网络宽度与候选容量写在 ZIP 内配置中。修改特征、模型或动作语义后，必须重新训练并导出配套源码和权重。

## 推荐修改顺序

先修改 `features.py` 增强可见信息，再修改 `model.py` 与配置。保持输入字段、动作 mask 和模型输出维度一致。训练可用较新的 PyTorch；凡是会进入 Bot 的模型、特征、动作和协议代码，都必须满足 Python 3.6 / Torch 1.8 的语法和 API 以适配 Botzone。

更换算法时需要调整 `policy.py` 与 `train.py`，复用 `env.py`。PPO learner 通过 `act`、`values`、`evaluate_actions` 接口工作，不依赖游戏动作细节。修改算法不是简单改配置名字；若使用 off-policy 算法，需要相应 replay、目标网络和采样语义。框架的架构、代码组织、组件职责和场景支持能力统一见 [框架架构与使用指南](rl_framework/docs/README.md)。

不要让特征访问 GameCore 的其他玩家手牌、未摸牌墙或底牌。`view(player)` 或麻将的私有事件观察器是特征边界。训练与 Bot 必须调用同一个编码函数，避免训练时使用平台看不到的信息。

## 注意点

这是示例代码，只验证了训练和部署闭环，不保证 AI 水平。麻将仅使用门风、圈风、手牌六通道作为网络输入；规则 mask 使用合法公开信息。双升/掼蛋使用状态和动作编码，默认最多保留 128 个候选动作；超过容量会确定性抽样并记录次数。

候选动作并非完整动作空间：相同牌型通常只保留一个具体案例；双升不主动甩牌，但支持按结构跟对手甩牌，连续对子的候选动作也只保留生成器给出的子集。提升候选动作覆盖、公开历史与团队/庄家信息编码，是推荐的作业改进方向。罕见复杂甩牌的候选动作生成仍可能较慢，需要结合 Botzone 时限继续优化。

双升/掼蛋单局训练，不模拟跨局升级赛。开局阶段策略与单局初始状态见各游戏 README。评估使用固定种子、轮换座位和随机合法动作对手，只提供低成本诊断；不能据此宣称 Bot 强度或稳定提升。
