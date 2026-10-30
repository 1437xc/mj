"""Exercise platform compound responses against independent judge event observers."""

import unittest
from unittest.mock import patch
import numpy as np
from mahjong.gamecore.engine import GameCore
from mahjong.observation import SeatView
from mahjong.botzone.session import Session
from mahjong.features import encode


def platform_messages(events, seat):
    messages = []
    i = 0
    while i < len(events):
        t = events[i].split()
        i += 1
        if t[0] == "Wind":
            messages.append("0 %d %s" % (seat, t[1]))
        elif t[0] == "Deal":
            messages.append("1 0 0 0 0 " + " ".join(t[1:]))
        elif t[0] == "Draw":
            messages.append("2 " + t[1])
        elif t[0] == "Player":
            p, event = t[1], t[2]
            if event in ("Peng", "Chi"):
                following = events[i].split()
                i += 1
                if following[:3] != ["Player", p, "Play"]:
                    raise AssertionError("Missing compound discard")
                messages.append(
                    "3 %s %s %s" % (p, event.upper(), " ".join(t[3:] + following[3:]))
                )
            elif event == "AnGang":
                messages.append("3 %s GANG" % p)
            elif event in ("Draw", "Play", "Gang", "BuGang"):
                messages.append(" ".join(["3", p, event.upper()] + t[3:]))
            elif event not in ("Hu", "Invalid"):
                raise AssertionError(event)
    return messages


class MahjongProtocolTest(unittest.TestCase):
    def test_self_draw_fourth_tile_counted_once(self):
        core = GameCore()
        core.r.seed(0)
        core.reset()
        core.shownTiles[core.curTile] = 3
        with patch(
            "mahjong.gamecore.engine.MahjongFanCalculator",
            return_value=[(8, 1, "test", "test")],
        ) as calculator:
            core.step({core.agent_names[core.curPlayer]: "Hu"})
        self.assertTrue(calculator.call_args.kwargs["is4thTile"])
        self.assertIsNone(core.invalid_info)

    def test_complete_protocol_games(self):
        compound = 0
        for seed in range(40):
            rng = np.random.RandomState(seed)

            def choose(arrays):
                legal = np.flatnonzero(arrays["action_mask"])
                return 1 if 1 in legal else int(rng.choice(legal))

            core = GameCore()
            core.r.seed(seed)
            core.reset()
            bots = [Session(choose) for _ in range(4)]
            observers = [SeatView(i) for i in range(4)]
            offsets = [0] * 4
            discards = {}
            steps = 0
            while not core.done:
                if core.state == 0:
                    # Platform already included this immediate discard in CHI/PENG.
                    core.step(
                        {
                            core.agent_names[core.curPlayer]: "Play "
                            + discards[core.curPlayer]
                        }
                    )
                responses = {}
                for p in range(4):
                    events = core.agents[p].events[offsets[p] :]
                    offsets[p] += len(events)
                    for event in events:
                        observers[p].request2obs(event)
                    for message in platform_messages(events, p):
                        responses[p] = bots[p].handle(message)
                    self.assertEqual(
                        sorted(bots[p].view.hand), sorted(observers[p].hand)
                    )
                    self.assertEqual(bots[p].view.packs, observers[p].packs)
                    self.assertEqual(bots[p].view.tileWall, observers[p].tileWall)
                    for key, value in encode(observers[p]).items():
                        # Nonactive observers may retain different obsolete legal sets;
                        # only active decision masks are meaningful.
                        if p in core.obs:
                            np.testing.assert_array_equal(
                                encode(bots[p].view)[key], value
                            )
                commands = {}
                for p in core.obs:
                    response = responses[p].split()
                    if response[0] in ("CHI", "PENG"):
                        compound += 1
                        discards[p] = response[-1]
                        response = response[:-1]
                    command = response[0].title()
                    if response[0] == "BUGANG":
                        command = "BuGang"
                    commands[core.agent_names[p]] = " ".join([command] + response[1:])
                core.step(commands)
                steps += 1
                if core.reward:
                    self.assertFalse(min(core.reward) == -30 and max(core.reward) == 10)
                self.assertLess(steps, 500)
        self.assertGreater(compound, 0)


if __name__ == "__main__":
    unittest.main()
