"""Student extension point: encode a private SeatView using NumPy only."""

import numpy as np

OBS_SHAPE = (6, 4, 9)
ACTION_COUNT = 235


def encode(view):
    observation = np.zeros((6, 36), dtype=np.float32)
    observation[0, view.OFFSET_TILE["F%d" % (view.seatWind + 1)]] = 1
    observation[1, view.OFFSET_TILE["F%d" % (view.prevalentWind + 1)]] = 1
    for tile in set(view.hand):
        observation[2 : 2 + view.hand.count(tile), view.OFFSET_TILE[tile]] = 1
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    mask[view.valid] = True
    return {"obs": observation.reshape(OBS_SHAPE), "action_mask": mask}
