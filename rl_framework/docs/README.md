# RLFramework：架构与使用指南

RLFramework 是本次课程作业使用的 PyTorch / TensorDict 强化学习组件库。它将环境交互、特征编码、策略决策、轨迹采样、回报计算和参数优化分开，使不同游戏能够复用训练流程，同时保留各自的规则、观测和动作表示。

本作业的国标麻将、双升和掼蛋均采用**同步整局 PPO**：多个 CPU actor 使用同一轮参数完成对局，主进程汇总数据后更新策略，再开始下一轮。四个座位共享策略参数，但各自保留独立的时间序列。双升和掼蛋只学习出牌阶段，准备阶段由游戏代码中的规则策略执行。

本文同时介绍框架的通用能力和课程代码的具体用法。框架接口可承载的场景，不等于每个内置算法都已提供对应模型、环境或训练调度；具体边界见“场景支持与选型”。

## 阅读导航

- [安装与课程入口](#安装与课程入口)
- [整体架构与数据流](#整体架构与数据流)
- [代码组织](#代码组织)
- [数据契约](#数据契约)
- [环境与特征](#环境与特征)
- [策略与动作](#策略与动作)
- [采样、路由与奖励](#采样路由与奖励)
- [回报计算与循环序列](#回报计算与循环序列)
- [学习器与算法](#学习器与算法)
- [工作流与参数存储](#工作流与参数存储)
- [课程中的组合示例](#课程中的组合示例)
- [场景支持与选型](#场景支持与选型)
- [扩展与排查](#扩展与排查)

## 安装与课程入口

在作业包根目录执行，训练环境建议使用 Python 3.10：

```bash
python -m pip install -r requirements.txt
python -m pip install -e ./rl_framework
python -m mahjong.train --output runs/mahjong_first --iterations 3 --actors 2
```

`requirements.txt` 统一描述训练依赖，包括 PyTorch 2.3.1、TensorDict 0.1.0、NumPy 1.26.4 和 PyMahjongGB。麻将算番通过安装 PyMahjongGB 使用，包内不附带其源码。框架自身要求 Python 3.10 及以上；课程依赖组合以根目录 requirements 为准。

将命令中的 `mahjong` 换成 `tractor` 或 `guandan` 即可训练另外两个游戏。训练输出目录应使用尚不存在的新目录。训练、恢复、评估和 Bot 导出的完整命令见[作业包使用说明](../../README.md)，游戏细节见[麻将](../../mahjong/README.md)、[双升](../../tractor/README.md)、[掼蛋](../../guandan/README.md)。

Bot 推理入口复用游戏的特征编码和普通 PyTorch 模型，不依赖框架的采样器、学习器和 TensorDict。部署方法以各游戏说明为准，训练 checkpoint 与导出的部署权重用途不同。

## 整体架构与数据流

框架围绕“谁产生数据、谁解释动作、谁更新参数”划分职责：

```text
游戏规则 / 外部环境
        │ reset / step
        ▼
MultiEnv ── 具名观测、奖励、结束标志 ──► Collector
   ▲                                      │
   │ 环境动作                             ├─ Router：agent → policy / version
   └──────────────────────────────────────┤
                                          └─ Policy.act：模型前向与动作分布
                                                   │
                                                   ▼
                                 Collection：相互独立的 agent 轨迹
                                                   │
                      ┌────────────────────────────┴─────────────────────┐
                      ▼                                                  ▼
         On-policy：逐轨迹计算 GAE                           Off-policy：写入 ReplayStore
                      │                                                  │
          按 policy 合并 / 构建循环序列                              抽样 transition
                      │                                                  │
                      └──────────────────► Learner.update ◄───────────────┘
                                                 │
                                            优化模型参数
                                                 │
                              On-policy workflow 发布 WeightStore 版本
                                                 │
                                            下一轮采样
```

环境和游戏模型不知道 PPO 的损失函数；学习器不知道“吃碰杠”或“某种牌型”的具体含义。策略负责把模型输出解释为动作及其概率，采样器负责把交互组织成正确的时间序列，工作流负责安排这些组件的执行顺序。

课程的并行采样在上述 Collector 外再增加一层 `ParallelEpisodeCollector`：主进程持有 learner 和权重，子进程持有本地环境、策略副本及 `SparseDecisionCollector`。并行的是对局采样，参数更新仍由一个 learner 统一执行。

## 代码组织

以下路径相对于 `rl_framework/`。顶层 Python 模块定义核心契约和数据类型，功能子目录放置具体实现。

```text
rl_framework/
├── pyproject.toml              # 安装信息与框架依赖
├── README.md                   # 文档入口
├── docs/README.md              # 本指南
├── tests/                      # 组件及组合行为测试
└── rlframework/
    ├── __init__.py             # 核心契约与数据类型的公开导出
    ├── data.py                 # 字段规范、轨迹、采样结果、循环批次
    ├── environment.py          # MultiEnv、AgentStep、MultiEnvStep
    ├── feature.py              # FeatureAdapter 及数组/函数适配器
    ├── policy.py               # Policy、ActorCriticPolicy 及输出类型
    ├── collector.py            # PolicyRef、RewardResolver
    ├── learner.py              # BatchLearner 协议
    ├── replay_store.py         # ReplayStore 接口
    ├── weight_store.py         # WeightStore、VersionedWeights
    ├── algorithm/
    │   ├── ppo.py              # PPO 配置、参考模型/策略、目标与学习器
    │   ├── a2c.py / a3c.py     # A2C 与共享参数 A3C-style 组件
    │   ├── pg.py / dqn.py      # Monte Carlo PG 与 DQN
    │   ├── ddpg.py / sac.py    # 连续动作算法
    │   ├── named_policy.py    # 具名多输入模型的类别策略
    │   ├── recurrent.py       # GRU 模型及循环类别策略
    │   └── distributions.py   # 合法动作掩码与条件类别分布
    ├── collectors/
    │   ├── sparse.py          # 稀疏决策、轮流行动采样
    │   ├── synchronous.py     # 同步联合行动采样
    │   ├── parallel.py        # 单机多进程整局采样
    │   ├── routing.py         # 智能体到策略的静态路由
    │   └── rewards.py         # 逐智能体奖励读取与分配
    ├── environments/gym.py    # 单智能体 Gym / Gymnasium 环境适配
    ├── processing/
    │   ├── gae.py             # 逐轨迹优势与回报计算
    │   └── sequence.py        # 序列切片、padding、burn-in 训练视图
    ├── replay/
    │   ├── memory.py          # 内存经验回放
    │   └── remote.py          # 远程回放客户端与服务注册
    ├── weight/
    │   ├── memory.py          # 内存参数版本存储
    │   └── remote.py          # 远程参数客户端与服务注册
    ├── workflow/
    │   ├── on_policy.py       # 多策略采样、处理、学习与发布
    │   └── off_policy.py      # 本地采样、入库与抽样学习
    └── utils/
        ├── tensor.py         # TensorDict 转换、CPU 复制与记录堆叠
        ├── module.py         # 模型设备与参数导入导出
        ├── rpc.py            # TensorDict 编解码与 TCP 通信
        └── fdl_bridge.py     # 可选的 FDL 特征编译和加载工具
```

`feature.py` 同时包含接口和两个轻量实现，便于直接接入手写特征。采样器、处理器和工作流通过所需方法组合，不要求继承统一的 `Collector`、`Processor` 或 `Workflow` 基类。

推荐从定义模块导入具体实现，使代码能直接反映组件来源：

```python
from rlframework.policy import ActorCriticPolicy
from rlframework.data import Collection, SequenceBatch
from rlframework.feature import ArrayFeatureAdapter
from rlframework.algorithm.named_policy import NamedCategoricalPolicy
from rlframework.collectors.sparse import SparseDecisionCollector
from rlframework.processing.gae import GAEProcessor
from rlframework.workflow.on_policy import MultiPolicyOnPolicyWorkflow, OnPolicyStack
```

接口层不依赖具体游戏和算法。`utils/__init__.py` 不主动导入工具，普通 `import rlframework` 不会加载 FDL 桥接、RPC 或具体训练流程。可选环境和特征工具的依赖由使用方单独提供。

## 数据契约

### 字段与批次

`TensorDict` 是带显式批次维度的张量字典，也允许嵌套字段。单条观测和单条 transition 的 `batch_size=[]`，一条轨迹为 `[T]`，普通学习批次为 `[B]`，循环学习批次为 `[B,T]`。特征维度位于批次维度之后，不能把特征长度当作样本数。

`FieldSpec(shape, dtype, semantic)` 声明一个字段的形状和类型；`shape` 不包含批次维。`TensorDictSpec` 声明必需字段并检查实际数据。`NestedKey` 可为字符串或字符串元组，例如 `("critic", "global_state")`，因此无需把所有信息拍平成一个向量。

### 轨迹与采样结果

| 类型 | 内容与职责 |
| --- | --- |
| `TrajectoryKey` | `env_id`、`episode_id`、`agent_id`、`segment_id`，唯一标识一个智能体的一段连续经历 |
| `Trajectory` | 上述标识、`policy_id`、`policy_version` 与按时间排列的 `[T]` TensorDict |
| `Collection` | 多条轨迹，以及本次实际 `env_steps`、`agent_steps`、`completed_episodes` |
| `SequenceBatch` | `[B,T]` 数据、有效位置 mask、初始循环状态和 burn-in 前缀 |

**策略共享不等于轨迹共享。** 四个座位即使使用同一个网络，其奖励递推和 RNN 状态也分别处理；先计算各自的回报，再把同一策略的数据合并用于优化。`Collection.records(policy_id)` 将指定策略的轨迹展平为 transition 列表，适合写入 replay，不能在 GAE 之前用它混合智能体的时间序列。

主要 transition 字段如下，表中的点号表示 TensorDict 的嵌套路径：

| 字段 | 含义 |
| --- | --- |
| `obs`、`action`、`next.obs` | 当前具名观测、完整动作树、该智能体的下一观测 |
| `reward` | 该决策对应的奖励；稀疏采样时为决策间隔内累计奖励 |
| `terminated`、`truncated` | 自然终止和时间限制等截断，决定 bootstrap 与递推边界 |
| `episode_start` | 该智能体的新回合起点，用于状态重置和序列边界 |
| `discount` | 下一状态价值的折扣系数 |
| `trace_discount` | 优势向后递推的折扣系数 |
| `policy.version` | 产生该决策的行为策略版本 |
| `policy.extras.logp`、`policy.extras.value` | PPO 等方法需要的采样时动作 log-probability 和价值估计 |
| `state_in`、`state_out` | 循环策略的输入和输出状态；无状态策略无需这些字段 |

环境、采样结果、回放和通信边界以 CPU 数据传递；学习器按配置将训练批次移动到模型设备。自定义策略也应正确处理设备转换。

## 环境与特征

### MultiEnv：统一环境入口

`MultiEnv` 定义两个方法：

```text
reset(config=None) -> MultiEnvStep
step(actions) -> MultiEnvStep
```

`actions` 为 `agent_id → 动作 TensorDict` 的映射。`MultiEnvStep.agents` 包含本次返回的智能体结果，每个 `AgentStep` 具有 `observation`、`reward`、`terminated`、`truncated` 和 `info`；`MultiEnvStep.shared` 放共享信息，例如所有座位的奖励向量。`episode_done` 表示整局结束，单个智能体退出不应提前结束其他智能体的轨迹。

同步场景需要为每个刚执行动作的智能体返回结果；轮流行动场景允许等该智能体下一次决策时才给出新的观测。时间限制截断必须给出真实末帧，不能使用 reset 后的首帧或上一条观测替代。

`GymMultiEnvAdapter` 把单智能体环境映射为 `agent_0`，并将动作树中的 `action` 叶子转换成标量或数组。需要区分终止与截断时，应使用返回两种独立标志的环境接口；适配器本身不安装或创建 Gym 环境。

### FeatureAdapter：具名数组接入

`FeatureAdapter.encode(raw_observation)` 将原始状态转换成 TensorDict，`spec` 描述输出字段。`ArrayFeatureAdapter` 检查已经编码的数组或 TensorDict；`CallableFeatureAdapter` 为每个字段调用指定编码函数，转换到声明 dtype 后检查形状。

课程游戏的 `features.py` 实现手写 NumPy 编码，`env.py` 将其组织为训练观测。编码不依赖 learner，Bot 可以复用相同函数。若需要 CNN、多组实体特征或候选动作矩阵，只需保留相应字段和维度，再提供能解释这些字段的模型。

`utils/fdl_bridge.py` 提供可选 FDL 编译与加载入口。FDL 是独立工具，课程训练不依赖它；使用该桥接时需另外提供 FDL 运行库、生成文件及所需算子。编译和训练分开，加载阶段检查生成文件和字段，不会自动编译。

## 策略与动作

### Policy 与 ActorCriticPolicy

`Policy` 继承 `torch.nn.Module`，负责模型前向、动作选择和循环状态：

```text
act(observation, state=None, deterministic=False) -> PolicyOutput
initial_state(observation) -> TensorDict 或 None
```

`PolicyOutput.action` 是完整动作树；`extras` 记录学习器需要的采样时统计量；`next_state` 是下一循环状态。`act` 面向单次环境决策，`deterministic=True` 用于策略定义的确定性选择。无状态策略返回 `None` 状态即可。

`ActorCriticPolicy` 增加两个学习接口：

```text
values(observation, state=None, episode_start=None) -> Tensor
evaluate_actions(observation, actions, state=None, episode_start=None) -> ActionEvaluation
```

`ActionEvaluation` 包含联合动作的 `log_prob`、`entropy` 和 `value`，形状与观测批次一致，通常为 `[B]` 或 `[B,T]`。策略解释动作结构与合法性，PPO/A2C 的通用目标只使用这些结果。`values` 是独立的 critic 路径，不应为了计算末帧价值而要求末帧存在可选动作。

### 具名输入、动作掩码与候选动作

`NamedCategoricalPolicy(module, input_keys, mask_key="action_mask")` 依次取出指定字段送入普通 PyTorch 模型。模型应返回 `[..., A]` 的 logits 和 `[...]` 的 value；mask 与 logits 的动作维必须对应，每条活动观测至少有一个合法动作。

麻将模型对固定动作集合输出分数。双升、掼蛋模型同时读取状态与候选动作特征，对候选行打分；策略输出候选行索引，由环境还原成真实出牌。候选集合由游戏代码生成，容量上限与 padding 也由游戏代码负责；框架不会自动枚举所有合法组合。

`ConditionalCategorical` 支持父类别和一个可选条件子类别。非激活分支不计入联合 log-probability，熵按条件分布加权。更深的动作树、混合离散/连续动作，需要自定义策略及其联合概率计算。

## 采样、路由与奖励

### 组件分工

| 组件 | 职能与适用情况 |
| --- | --- |
| `PolicyRef` | 将策略 ID、策略实例和行为版本绑定在一起 |
| `StaticPolicyRouter` | 按 agent ID 选择策略；同一 policy ID 对应同一策略实例 |
| `RewardResolver` | 为指定智能体读取一次环境推进产生的奖励 |
| `SharedTensorRewardResolver` | 从 `MultiEnvStep.shared` 的奖励向量按座位索引取值，可设置缩放 |
| `SynchronousStepCollector` | 每次联合行动后闭合各行动者的 transition，适合同步控制 |
| `SparseDecisionCollector` | 等待同一智能体的下次观测或结束再闭合 transition，适合轮流行动 |
| `ParallelEpisodeCollector` | 多进程运行本地采样器，等待整局完成后汇总 Collection |

奖励 resolver 负责“奖励属于谁”，collector 负责“何时累加及如何折扣”。轮流行动时，非当前行动者也可能获得奖励，不能只把结算奖励写给最后出牌者。课程通过共享奖励向量让每个座位获得对应奖励。

### 稀疏决策的时间语义

稀疏 collector 保留尚未闭合的决策：某座位行动后，其他座位可以继续行动，直到它再次决策或结束时才形成完整记录。

- 默认 `discount_clock="decision"`：对间隔内奖励求和，每次该智能体决策折扣一次。
- `discount_clock="env_step"`：对间隔内每个环境步折扣。间隔为 `k` 时，bootstrap 系数为 `gamma**k`，优势 trace 系数为 `gamma**k * gae_lambda`；lambda 仍按智能体决策计一次。

本地采样器的 `collect_episode()` 完成当前局，没有进行中的局才 reset；`collect(min_env_steps)` 保留活动环境和循环状态，不把预算边界伪装为环境终止。稀疏采样可能留有 pending 决策，相关参数不能在决策未闭合时改变。on-policy 工作流会先完成仍有 pending 的局再更新，因此实际采样量可能超过步预算。

### 单机并行整局采样

`ParallelEpisodeCollector` 使用 `spawn` 创建常驻 actor。工厂 `factory(worker_id)` 必须可序列化，并返回具有 `router`、`environment` 和 `collect_episode` 的本地采样器；每个 actor 使用不同的 `env_id`。

每轮将主进程的 CPU 参数快照和版本发给 actor。所有 actor 完成指定局数后，主进程检查轨迹标识与版本，再进行学习。`actors × episodes_per_actor` 决定每轮采样局数；整局耗时不一致时需要等待较慢的 actor。它不提供局中途切换参数或异步更新模式，属性 `synchronous=False` 是采样接口中的稀疏标志，不代表异步 PPO。

`episode_cursor` 决定下一局种子序号，训练恢复时应保存和还原。接口不保存各 actor 的动作采样 RNG 状态，因此恢复训练不保证逐位重现。`actor_pids` 和 `last_actor_stats` 可用于诊断；异常携带 worker 信息，超时回收该采样器创建的进程。使用 `with` 或显式 `close()` 释放资源。

## 回报计算与循环序列

### GAEProcessor

`GAEProcessor.process(trajectory, policy)` 对一条独立轨迹添加 `advantage` 和 `return`。它在参数更新前使用采样时 value 和下一状态价值计算：

```text
delta_t = reward_t + discount_t × V(next_obs_t) − V(obs_t)
advantage_t = delta_t + trace_discount_t × advantage_(t+1)
return_t = advantage_t + V(obs_t)
```

轨迹自带的 discount 字段优先，保证计算沿用采样器的决策时钟。自然终止不 bootstrap；时间截断允许用真实末帧 bootstrap，但不向下一局递推。采样片段末尾同样结束本片段的优势递推。循环策略 bootstrap 使用该决策的 `state_out`。

### SequenceBatcher

前馈策略可在逐轨迹计算 GAE 后按策略直接拼接。循环策略使用 `SequenceBatcher` 保持时序：默认保留完整轨迹，也可配置 `unroll_length=32, burn_in=8` 生成训练窗口与状态预热前缀。

`SequenceBatch.valid_mask` 排除 padding；padding 采用真实观测副本保留有效动作结构，但不进入损失和优势归一化。burn-in 使用当前权重无梯度重算输入状态，也不计入训练样本分母。训练时只打乱序列行，不打乱序列内部时间步。

`GRUActorCritic` 与 `RecurrentCategoricalPolicy` 提供参考实现。自定义循环策略需明确初始状态、下一状态和批次时间轴语义，不能只替换一个网络层就假定采样与训练状态已自动衔接。

## 学习器与算法

`BatchLearner` 约定 `update(batch) -> 指标映射`。学习器拥有优化器及相应算法目标，不负责环境交互、对局种子或日志落盘。

| 算法组件 | 主要能力 | 组合要求 |
| --- | --- | --- |
| `PPOLearner` / `PPOConfig` | clipped surrogate、value loss、entropy、多 epoch / minibatch；支持普通和循环批次 | `ActorCriticPolicy`、采样时 logp/value、GAE 结果 |
| `A2CLearner` | 共用 actor-critic 评估接口和优势数据进行更新 | 按其配置提供 policy 与已处理批次 |
| `A3CSharedLearner` | worker 本地求梯度，加锁更新共享参数，再同步 worker | 使用 `sync_to` / `update_from`，自行安排 worker；不直接充当课程 workflow 的 learner |
| `PGLearner` / `MonteCarloReturnProcessor` | 离散动作 Monte Carlo policy gradient | 自行组合整局回报处理和更新循环 |
| `DQNLearner` | 离散动作 Q 学习、目标网络与合法动作处理 | replay 与对应 DQN policy / 网络 |
| `DDPGLearner` | 连续动作确定性 actor、Q critic 和目标网络 | replay、探索噪声和动作尺度适配 |
| `SACLearner` | 连续动作 tanh Gaussian、双 Q 与固定温度项 | replay；温度由配置给定 |

`PPOPolicy` / `MLPActorCritic` 是向量观测、离散动作的参考组合；课程使用的是 `NamedCategoricalPolicy` 加各游戏自己的模型。PPO learner 通过 `evaluate_actions` 工作，因此更换 CNN、候选动作评分网络或循环策略时，可继续复用同一优化目标。

DDPG/SAC 的内置动作表示归一化到 `[-1,1]`，环境范围不同需要适配。PG、DQN、DDPG、SAC 的参考实现有各自的字段和模型约定，不能仅更换算法名称就套用课程 PPO 配置；循环、结构化动作和多策略支持应逐项核对对应 learner。

PPO 指标包括策略/价值损失、entropy、近似 KL、clip fraction 和概率比。工作流中的 `gradient_updates` 统计实际优化器步数；一次 `learner.update` 可能包含多次优化，不能把轮数等同于梯度更新次数。

## 工作流与参数存储

### MultiPolicyOnPolicyWorkflow

`OnPolicyStack(policy, processor, learner, batcher=None)` 组织一套策略的处理和学习组件。`MultiPolicyOnPolicyWorkflow` 一次 `step()` 的顺序为：

1. 联合采样，必要时闭合稀疏 pending 决策。
2. 对所有策略下的每条独立轨迹计算 GAE 和 bootstrap，此时尚未更新参数。
3. 按 `policy_id` 分组，拼接前馈批次或生成循环批次。
4. 对每个有数据的 stack 调用一次 `learner.update`。
5. 发布该策略的新权重版本，返回本轮指标。

同一参数共享组使用一个 policy ID、一个 stack 和一个优化器。不同 stack 可以拥有不同网络和观测形状，但不允许共享 `nn.Parameter` 对象；同一 stack 内的数据仍须可合并。路由和 stack 的策略 ID 必须完全对应，learner 必须拥有路由所指的同一策略实例，采样器与 GAE 的折扣配置也必须一致。

本地采样可用 `step(min_env_steps=...)` 指定步预算，或用 `episode_configs=[...]` 组织多次整局采样；`on_episode` 回调对应列表中的采样结果。课程并行采样器通过 `actors` 和 `episodes_per_actor` 控制整局预算，直接调用 `step()`。`last_collection` 保留最近一次真实采样结果。

`env_steps` 统计联合环境推进次数，`agent_steps` 统计实际动作数，`completed_episodes` 统计完成局数。在稀疏采样中，动作与其 transition 闭合可能发生在不同批次，因此动作数不必总等于本批记录数。

### ReplayStore 与 off-policy 工作流

`ReplayStore` 定义 `add(records)`、`sample(batch_size)` 和 `__len__()`。`InMemoryReplayBuffer` 保存 transition，`RemoteReplayStoreClient` 使用同一接口访问远程服务。

`LocalOffPolicyWorkflow` 完成一局后，提取指定策略的记录写入 replay；样本量足够时，按 `updates_per_episode` 抽样调用 learner。目标网络更新由算法 learner 负责。这个工作流不自动协调远程 actor 的参数同步，也不内置多策略 off-policy 调度。

### WeightStore 与远程边界

`WeightStore.publish(policy_id, parameters)` 发布参数并返回带递增版本的 `VersionedWeights`，`latest(policy_id)` 读取最新版本。内存和远程实现可通过相同接口替换。它保存的是最新参数，不是模型检查点、历史对手池或完整训练状态；课程 checkpoint 由 `train.py` 负责。

远程 replay / weight 使用 `utils.rpc` 的 TCP 服务，传输 JSON 元数据和 tensor bytes。课程单机 actor 使用本地进程 Pipe 传输序列化 Collection，两者是不同通道。TCP 组件可用于自行组装跨机 actor–learner，但不包含集群启动、自动重试、认证、TLS 或容错调度。

## 课程中的组合示例

每个游戏都按如下职责拆分，阅读顺序建议为 `env.py → policy.py → train.py`，再看模型与特征：

| 游戏文件 | 负责内容 |
| --- | --- |
| `gamecore/` | 规则、状态推进和结算 |
| `env.py` | 适配 MultiEnv、活动座位、动作还原和奖励 |
| `features.py` | 训练和 Bot 共用的观测编码 |
| `actions.py` / `heuristics.py` | 双升、掼蛋的候选出牌与准备阶段规则 |
| `model.py` | 普通 PyTorch 网络，返回动作分数和价值 |
| `policy.py` | NamedCategoricalPolicy、座位路由、actor 本地采样器工厂 |
| `train.py` | 并行整局 PPO 组装、配置、日志、保存和恢复 |
| `evaluate.py` / `export_bot.py` / `botzone/` | 评估、部署权重导出和平台交互 |

以下是从作业包根目录使用麻将组件构建一轮整局 PPO 的最小示例。将它保存为独立 Python 文件运行；多进程启动需要 `__main__` 保护。

```python
from functools import partial

from mahjong.policy import make_policy, make_collector
from rlframework.collector import PolicyRef
from rlframework.collectors.routing import StaticPolicyRouter
from rlframework.collectors.parallel import ParallelEpisodeCollector
from rlframework.algorithm.ppo import PPOConfig, PPOLearner
from rlframework.processing.gae import GAEProcessor
from rlframework.workflow.on_policy import OnPolicyStack, MultiPolicyOnPolicyWorkflow


def main():
    config = PPOConfig()
    policy = make_policy(hidden=64)
    learner = PPOLearner(policy, config, device="cpu")
    router = StaticPolicyRouter({
        str(seat): PolicyRef("shared", policy) for seat in range(4)
    })
    factory = partial(
        make_collector, hidden=64, gamma=config.gamma,
        gae_lambda=config.gae_lambda, seed=2026,
    )
    with ParallelEpisodeCollector(
        factory, router, actors=2, episodes_per_actor=1,
        gamma=config.gamma, gae_lambda=config.gae_lambda, seed=2026,
    ) as collector:
        workflow = MultiPolicyOnPolicyWorkflow(collector, {
            "shared": OnPolicyStack(
                policy, GAEProcessor(config.gamma, config.gae_lambda), learner,
            ),
        })
        print(workflow.step())


if __name__ == "__main__":
    main()
```

该例展示组件连接，不包含训练日志、评估和 checkpoint 保存。实际作业训练应使用各游戏的 `train.py`，并保留 `model.py`、`features.py` 与 Bot 推理的一致性。

## 场景支持与选型

下表区分可直接使用的组件和仍需编写的任务适配；它描述接口与实现能力，不代表所有组合都已验证收敛。

| 场景 | 框架提供的支持 | 使用方需要完成的工作或限制 |
| --- | --- | --- |
| 单智能体向量观测 | GymMultiEnvAdapter、MLP 参考策略与学习器 | 提供环境，匹配观测/动作范围及依赖 |
| 多智能体同步行动 | SynchronousStepCollector、逐智能体轨迹 | 环境为每个行动者返回结果，正确表达死亡与整局结束 |
| 轮流行动、稀疏决策 | SparseDecisionCollector、独立奖励 resolver | 明确活动座位、奖励归属与折扣时钟；课程三游戏采用此路径 |
| 多智能体参数共享 | 同一 policy ID / stack 服务多个 agent | 共享组内观测和动作结构须一致，时间序列仍分开 |
| 多策略或异构智能体 | 分策略 stack、独立模型与更新 | 每种策略提供相应接口；不自动将不同形状拼到一起，不允许跨 stack 共享参数 |
| CNN、多字段、实体或候选矩阵输入 | 嵌套 TensorDict、特征适配器、NamedCategoricalPolicy | 自定义编码和模型；变长输入需要显式 padding/mask 或适配策略 |
| 合法动作约束和候选动作 | 类别策略 mask、候选评分模型接口 | 候选生成/裁剪由游戏负责；课程候选子集不保证覆盖全部合法出牌组合 |
| 条件或结构化动作 | 动作树、联合动作评估接口、两级 ConditionalCategorical | 更深结构、混合动作需自定义分布和策略；不是全部算法的即插即用能力 |
| 部分可观测、循环记忆 | RNN 状态采样、GRU 参考策略、SequenceBatcher、burn-in | 设计历史信息与状态；课程默认模型为前馈网络，需主动接入循环策略与 batcher |
| 集中训练、分散执行 | ActorCriticPolicy 可分别解释 actor/critic 字段，环境可提供共享信息 | 自行把全局信息写入训练观测并提供 critic；shared 不会自动成为 critic 输入，不能让执行 actor 泄漏不可见信息 |
| 连续动作 | DDPG / SAC 模型、策略、learner 与 replay | 环境动作缩放和相应训练组合；内置 PPOPolicy 是类别策略，连续 PPO 要另外实现策略接口 |
| 单机并行 CPU 采样、可选 GPU 学习 | ParallelEpisodeCollector；learner 的 device 参数 | actor 使用 CPU；GPU 学习需可用 CUDA 和正确设备转换，不包含多 GPU 梯度同步 |
| 跨机存储与参数通信 | 远程 ReplayStore / WeightStore、TCP RPC | 自行安排进程启动、发布/拉取时机和故障处理；不是完整分布式训练系统 |
| 自博弈与对手管理 | 座位路由、共享策略可用于同版本自博弈 | 历史对手池、联赛、冻结对手轮换与自动课程调度需另行实现 |

## 扩展与排查

扩展作业通常按以下顺序进行：

1. **先定义信息与动作。** 列出可见信息、字段形状、合法动作以及奖励结算时机。特别区分环境推进一步与某个座位决策一次。
2. **实现环境和特征。** 检查 reset、自然终止、截断末帧，以及共享奖励是否分配给每个相关智能体。
3. **实现模型和策略。** 普通离散或候选动作优先使用 `NamedCategoricalPolicy`；特殊动作或循环状态实现相应策略接口，确保采样和重评估的概率含义一致。
4. **选择采样和学习组合。** 轮流行动用稀疏采样，同步行动用同步采样；增加循环网络时配套 SequenceBatcher，更换 off-policy 算法时配套 replay 与对应调度。
5. **用小规模完整对局检查。** 查看样本数、策略版本、合法动作、有限数值、梯度更新和参数变化，再扩大 actor 数或训练时长。

常见问题与检查位置：

| 现象 | 优先检查 |
| --- | --- |
| 活动观测没有合法动作 | env 的活动座位判断、候选生成和 `action_mask`；终局价值计算应走 `values` |
| 同一局出现重复轨迹或种子 | actor 的 `env_id`、episode cursor 和工厂种子设置 |
| GAE 或 RNN 跨座位混合 | 是否先按独立 Trajectory 处理，再按 policy 合并 |
| 截断后的价值异常 | 是否提供真实末帧、正确区分 `terminated` / `truncated` |
| 多策略工作流初始化报错 | 路由 ID、policy 实例、learner 归属、参数共享以及 gamma/lambda 是否一致 |
| 增加 actor 后变慢 | 每局耗时差异、CPU 线程、传输开销和等待屏障；并行不保证线性加速 |
| 训练能运行但 Bot 行为不同 | 特征、候选顺序、模型结构与导出权重是否一致 |

包内 `tests/` 可用来阅读组件行为示例：`test_collection_and_algorithms.py` 关注采样与算法，`test_core_reuse.py` 关注结构化动作、循环序列和多策略组合，`test_parallel.py` 关注整局并行，`test_remote.py` 关注远程存储。可选 FDL 相关测试需要额外的 FDL 库及示例资源，作业包本身不提供这些资源；课程训练入口不需要启用它们。
