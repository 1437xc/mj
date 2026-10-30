"""Single-machine, version-consistent whole-episode collection.

Workers own environments and CPU policy replicas; the calling process owns the
learner. Every request finishes whole episodes under one immutable snapshot.
Local pipes carry serialized CPU collections, not live shared tensor handles."""

from __future__ import annotations

import multiprocessing as mp
import pickle
import time
import traceback

from rlframework.data import Collection
from rlframework.utils.module import load_module_parameters, module_parameters


def _worker(connection, factory, worker_id):
    import torch

    torch.set_num_threads(1)
    collector = None
    try:
        collector = factory(worker_id)
        connection.send_bytes(pickle.dumps(("ready", None), protocol=4))
        while True:
            message = pickle.loads(connection.recv_bytes())
            if message is None:
                break
            snapshots, configs, deterministic = message
            for name, (version, parameters) in snapshots.items():
                load_module_parameters(collector.router.policies[name], parameters)
                collector.router.policies[name].eval()
                collector.router.set_version(name, version)
            batches = [
                collector.collect_episode(config=config, deterministic=deterministic)
                for config in configs
            ]
            stats = {
                "candidate_overflows": getattr(
                    collector.environment, "candidate_overflows", 0
                )
            }
            connection.send_bytes(pickle.dumps(("ok", (batches, stats)), protocol=4))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send_bytes(
                pickle.dumps(("error", traceback.format_exc()), protocol=4)
            )
        except (BrokenPipeError, OSError):
            pass
    finally:
        if collector is not None:
            close = getattr(collector.environment, "close", None)
            if close is not None:
                close()
        connection.close()


class ParallelEpisodeCollector:
    """Collector-compatible barrier over persistent spawned actor processes.

    ``factory(worker_id)`` must be picklable and construct a local collector.
    ``router`` belongs to the learner; its policy weights and published versions
    are copied at each call. Budgets count complete episodes, never partial ones.
    """

    synchronous = False

    def __init__(
        self,
        factory,
        router,
        *,
        actors=4,
        episodes_per_actor=1,
        gamma=0.99,
        gae_lambda=0.95,
        seed=0,
        timeout=300,
    ):
        if actors < 1 or episodes_per_actor < 1:
            raise ValueError("actors and episodes_per_actor must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.router, self.gamma, self.gae_lambda = router, gamma, gae_lambda
        self.episodes_per_actor, self.seed, self.timeout = (
            episodes_per_actor,
            seed,
            timeout,
        )
        self._episode = 0
        self._ready = False
        self.last_actor_stats = []
        self._workers, self._connections = [], []
        context = mp.get_context("spawn")
        try:
            for index in range(actors):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker,
                    args=(child, factory, index),
                    name="rl-actor-%d" % index,
                )
                process.start()
                child.close()
                self._workers.append(process)
                self._connections.append(parent)
        except BaseException:
            self.close()
            raise

    @property
    def has_pending(self):
        return False

    @property
    def actor_pids(self):
        return [p.pid for p in self._workers]

    @property
    def episode_cursor(self):
        """Next global episode number; persist this to avoid seed reuse on resume."""
        return self._episode

    @episode_cursor.setter
    def episode_cursor(self, value):
        if int(value) < 0:
            raise ValueError("episode cursor must be nonnegative")
        self._episode = int(value)

    def collect_episode(self, deterministic=False, config=None):
        if not self._workers:
            raise RuntimeError("Collector is closed")
        versions = self.router.versions
        snapshots = {
            name: (versions[name], module_parameters(policy))
            for name, policy in self.router.policies.items()
        }
        try:
            if not self._ready:
                # Wait before sending large snapshots, so a stuck factory cannot
                # block the parent's first pipe write indefinitely.
                deadline = time.monotonic() + self.timeout
                for connection in self._connections:
                    if not connection.poll(max(0, deadline - time.monotonic())):
                        raise TimeoutError("Actor initialization exceeded deadline")
                    status, payload = pickle.loads(connection.recv_bytes())
                    if status != "ready":
                        raise RuntimeError(
                            "Actor failed during initialization:\n" + str(payload)
                        )
                self._ready = True
            for connection in self._connections:
                configs = []
                for _ in range(self.episodes_per_actor):
                    item = dict(config or {})
                    item["seed"] = int(item.get("seed", self.seed)) + self._episode
                    configs.append(item)
                    self._episode += 1
                connection.send_bytes(
                    pickle.dumps((snapshots, configs, deterministic), protocol=4)
                )
            batches = []
            self.last_actor_stats = []
            deadline = time.monotonic() + self.timeout
            for connection in self._connections:
                if not connection.poll(max(0, deadline - time.monotonic())):
                    raise TimeoutError(
                        "Actor exceeded whole-episode collection deadline"
                    )
                status, payload = pickle.loads(connection.recv_bytes())
                if status != "ok":
                    raise RuntimeError("Actor failed:\n" + payload)
                worker_batches, stats = payload
                batches.extend(worker_batches)
                self.last_actor_stats.append(stats)
            trajectories = [t for batch in batches for t in batch.trajectories]
            identities = [t.key for t in trajectories]
            if len(set(identities)) != len(identities):
                raise ValueError("Actors returned duplicate trajectory identities")
            if any(t.policy_version != versions[t.policy_id] for t in trajectories):
                raise ValueError("Actor returned a stale behavior-policy version")
            return Collection(
                trajectories,
                sum(b.env_steps for b in batches),
                sum(b.agent_steps for b in batches),
                sum(b.completed_episodes for b in batches),
            )
        except BaseException:
            self.close()
            raise

    def close(self):
        """Stop only processes owned by this collector, including failed workers."""
        # Closing the parent ends wakes idle receivers without a potentially
        # blocking send to a worker that is itself blocked sending a large batch.
        for connection in self._connections:
            connection.close()
        for process in self._workers:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        self._workers, self._connections = [], []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
