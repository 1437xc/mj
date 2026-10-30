# 国标麻将示例

先阅读上级 README，把命令中的游戏名改为 `mahjong`。

规则：136 张无花复式国标麻将，八番起和；每个玩家有自己的 34 张牌墙。算番使用 `requirements.txt` 中的 PyMahjongGB，由 pip 安装最新版；Bot 直接使用 Botzone 预装库。`observation.py` 重建每个座位的手牌、公开副露和合法动作；网络看不到其他玩家手牌与牌墙。

`features.py` 输出 `obs[6,4,9]`：座风、圈风、四层手牌计数；`action_mask[235]` 与 `observation.py` 的固定动作编号一致。`model.py` 是两个卷积层加策略头、价值头。该极简特征不包含完整公开历史，学生可优先增强这些信息。

平台的 CHI/PENG 响应需要同时附带打牌。Bot 先选吃/碰，再在私有状态副本上选打牌，最后等待裁判广播后提交状态；训练使用两个依次决策。终局分数除以 32 作为训练奖励，荒庄为零。四个座位共享策略，仍然分别形成轨迹。

部署：`mahjong_bot.zip` + `mahjong_model.pt`；权重放个人存储根目录；选择 Python 3.6、简单交互、长时运行、136 张无花复式模式。重启时从 requests/responses 历史重建状态；本 Bot 不使用额外 data/globaldata。144 张与花牌规则不在本示例范围内。

可优先修改：`features.py`（公开历史、副露与剩余可见张数）、`model.py`、`configs/default.json`。规则 mask、CHI/PENG 复合响应和暗杠身份跟踪有协议约束，修改后应重跑 `tests/test_protocol.py`。
