"""Loopback tests for the self-hosted TensorDict TCP transport and weight store."""

import unittest

import torch
from tensordict import TensorDict

from rlframework.replay.memory import InMemoryReplayBuffer
from rlframework.replay.remote import RemoteReplayStoreClient, register_replay_store
from rlframework.utils.rpc import TensorDictRpcClient, TensorDictRpcServer
from rlframework.weight.memory import InMemoryWeightStore
from rlframework.weight.remote import RemoteWeightStoreClient, register_weight_store


class RemoteRuntimeTests(unittest.TestCase):
    def test_rpc_round_trip_and_remote_weight_versions(self):
        server = TensorDictRpcServer()

        def double(payload, metadata):
            response = payload.clone()
            response.set("obs", payload.get("obs") * 2)
            return response, {"worker": "loopback"}

        server.register("double", double)
        register_weight_store(server, InMemoryWeightStore())
        register_replay_store(server, InMemoryReplayBuffer(8, seed=0))
        server.start()
        try:
            host, port = server.address
            client = TensorDictRpcClient(host, port)
            request = TensorDict({"obs": torch.tensor([[1.0, 2.0]])}, batch_size=[1])
            response, metadata = client.request("double", request)
            self.assertEqual(metadata["worker"], "loopback")
            self.assertEqual(response.get("obs").tolist(), [[2.0, 4.0]])
            weights = RemoteWeightStoreClient(client)
            first = weights.publish(
                "policy", TensorDict({"weight": torch.tensor([1.0])}, batch_size=[])
            )
            second = weights.publish(
                "policy", TensorDict({"weight": torch.tensor([2.0])}, batch_size=[])
            )
            latest = weights.latest("policy")
            self.assertEqual((first.version, second.version, latest.version), (1, 2, 2))
            self.assertEqual(latest.parameters.get("weight").tolist(), [2.0])
            replay = RemoteReplayStoreClient(client)
            replay.add([TensorDict({"obs": torch.tensor([1.0])}, batch_size=[])])
            self.assertEqual(len(replay), 1)
            self.assertEqual(replay.sample(1).get("obs").tolist(), [[1.0]])
        finally:
            server.close()
