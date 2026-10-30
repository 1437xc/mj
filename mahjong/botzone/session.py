"""Botzone messages to the same private observer used by training.

CHI/PENG require an immediate discard in one response. Evaluate that second
decision on a private copy, committing only when the judge confirms the event.
"""

import copy
from ..observation import SeatView
from ..features import encode


class Session:
    def __init__(self, choose, capacity=None):
        self.choose = choose
        self.view = None
        self.drawn = False
        self.concealed = None

    def handle(self, request, forced=None):
        t = request.split()
        obs = None
        response = "PASS"
        if t[0] == "0":
            self.seat = int(t[1])
            self.view = SeatView(self.seat)
            self.view.request2obs("Wind " + t[2])
        elif t[0] == "1":
            if any(int(x) for x in t[1:5]):
                raise ValueError(
                    "This course Bot targets 136-tile duplicate Mahjong without flowers"
                )
            self.view.request2obs("Deal " + " ".join(t[5:18]))
        elif t[0] == "2":
            obs = self.view.request2obs("Draw " + t[1])
            self.drawn = True
        elif t[0] == "3":
            p, event = int(t[1]), t[2]
            if event == "DRAW":
                self.view.request2obs("Player %d Draw" % p)
                self.drawn = True
            elif event == "GANG":
                if self.drawn:
                    command = "Player %d AnGang" % p
                    if p == self.seat:
                        if self.concealed is None:
                            raise ValueError("Missing own concealed-kong identity")
                        command += " " + self.concealed
                    self.view.request2obs(command)
                else:
                    self.view.request2obs("Player %d Gang" % p)
            elif event == "BUGANG":
                obs = self.view.request2obs("Player %d BuGang %s" % (p, t[3]))
            elif event in ("PLAY", "PENG", "CHI"):
                self.drawn = False
                if event != "PLAY":
                    self.view.request2obs(
                        "Player %d %s%s"
                        % (p, event.title(), " " + t[3] if event == "CHI" else "")
                    )
                obs = self.view.request2obs("Player %d Play %s" % (p, t[-1]))
            else:
                raise ValueError("Unsupported Mahjong event " + event)
        else:
            raise ValueError("Unknown Mahjong request " + request)
        if forced is not None:
            response = forced
        elif obs is not None and obs.valid:
            command = obs.action2response(self.choose(encode(obs))).split()
            response = " ".join([command[0].upper()] + command[1:])
            if command[0] in ("Peng", "Chi"):
                temporary = copy.deepcopy(self.view)
                temporary.request2obs("Player %d " % self.seat + " ".join(command))
                discard = temporary.action2response(
                    self.choose(encode(temporary))
                ).split()
                if discard[0] != "Play":
                    raise ValueError("CHI/PENG follow-up must discard")
                response += " " + discard[1]
        if t[0] == "2" and response.startswith("GANG "):
            self.concealed = response.split()[1]
        return response
