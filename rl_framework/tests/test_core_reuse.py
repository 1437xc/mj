"""Numerical and temporal invariants of reusable actor-critic components."""

import copy
import unittest
import torch
from tensordict import TensorDict
from rlframework.algorithm.ppo import MLPActorCritic, PPOPolicy, PPOLearner, PPOConfig, ppo_objective
from rlframework.collector import PolicyRef
from rlframework.collectors.routing import StaticPolicyRouter
from rlframework.collectors.synchronous import SynchronousStepCollector
from rlframework.collectors.sparse import SparseDecisionCollector
from rlframework.algorithm.distributions import ConditionalCategorical
from rlframework.environment import AgentStep, MultiEnv, MultiEnvStep
from rlframework.policy import ActionEvaluation
from rlframework.processing.gae import GAEProcessor
from rlframework.algorithm.recurrent import GRUActorCritic, RecurrentCategoricalPolicy
from rlframework.processing.sequence import SequenceBatcher, training_view
from rlframework.workflow.on_policy import MultiPolicyOnPolicyWorkflow, OnPolicyStack
from test_collection_and_algorithms import CounterEnv, AlternatingEnv


def collector(policy, env=None, **kwargs):
    return SynchronousStepCollector(
        env or CounterEnv(5),
        StaticPolicyRouter({"agent": PolicyRef("main", policy)}),
        **kwargs,
    )


class ReuseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_step_rollouts_advance_reset_seeds(self):
        class SeededCounter(CounterEnv):
            def __init__(self):
                super().__init__(length=2)
                self.seeds = []

            def reset(self, config=None):
                self.seeds.append(config["seed"])
                return super().reset(config)

        environment = SeededCounter()
        policy = PPOPolicy(MLPActorCritic(2, 2))
        result = collector(policy, environment).collect(6, config={"seed": 40})
        self.assertEqual(environment.seeds, [40, 41, 42])
        self.assertEqual(result.completed_episodes, 3)

    def test_conditional_probability_entropy_and_inactive_gradient(self):
        parent = torch.tensor([0.2, -0.3], requires_grad=True)
        child = torch.tensor([0.7, -0.2, 0.1], requires_grad=True)
        dist = ConditionalCategorical(parent, child)
        probabilities = [
            dist.log_prob(
                TensorDict(
                    {"intent": torch.tensor(0), "direction": torch.tensor(i)}, []
                )
            ).exp()
            for i in range(3)
        ]
        inactive = TensorDict(
            {"intent": torch.tensor(1), "direction": torch.tensor(999)}, []
        )
        probabilities.append(dist.log_prob(inactive).exp())
        probabilities = torch.stack(probabilities)
        self.assertTrue(torch.allclose(probabilities.sum(), torch.tensor(1.0)))
        self.assertTrue(
            torch.allclose(dist.entropy(), -(probabilities * probabilities.log()).sum())
        )
        dist.log_prob(inactive).backward()
        self.assertEqual(float(child.grad.abs().sum()), 0.0)
        self.assertGreater(float(parent.grad.abs().sum()), 0)

    def test_conditional_illegal_child_and_unavailable_branch(self):
        dist = ConditionalCategorical(
            torch.zeros(2),
            torch.zeros(3),
            torch.tensor([False, True]),
            torch.zeros(3, dtype=torch.bool),
        )
        self.assertTrue(torch.isfinite(dist.entropy()))
        self.assertEqual(int(dist.sample().get("intent")), 1)
        with self.assertRaises(ValueError):
            ConditionalCategorical(
                torch.zeros(2),
                torch.zeros(3),
                child_mask=torch.zeros(3, dtype=torch.bool),
            )

    def test_fragments_preserve_episode_and_recurrent_state(self):
        policy = RecurrentCategoricalPolicy(GRUActorCritic(2, 2, 8))
        c = collector(policy)
        first, second = c.collect(2), c.collect(2)
        self.assertEqual(first.env_steps, 2)
        self.assertEqual(first.completed_episodes, 0)
        self.assertFalse(first.trajectories[0].data.get("truncated").any())
        self.assertEqual(
            first.trajectories[0].key.episode_id, second.trajectories[0].key.episode_id
        )
        self.assertFalse(second.trajectories[0].data.get("episode_start").any())
        self.assertTrue(
            torch.equal(
                first.trajectories[0].data.get(("state_out", "hidden"))[-1],
                second.trajectories[0].data.get(("state_in", "hidden"))[0],
            )
        )
        self.assertEqual(c.collect_episode().completed_episodes, 1)
        self.assertTrue(
            bool(c.collect_episode().trajectories[0].data.get("episode_start")[0])
        )

    def test_recurrent_step_unroll_and_reset_agree(self):
        policy = RecurrentCategoricalPolicy(GRUActorCritic(2, 2, 8))
        data = collector(policy).collect_episode().trajectories[0].data
        seq = SequenceBatcher().batch([data])
        items, mask, kwargs = training_view(seq, policy)
        evaluation = policy.evaluate_actions(
            items.get("obs"), items.get("action"), **kwargs
        )
        self.assertTrue(
            torch.allclose(
                evaluation.log_prob[0],
                data.get(("policy", "extras", "logp")),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                evaluation.value[0], data.get(("policy", "extras", "value")), atol=1e-6
            )
        )
        poisoned = TensorDict({"hidden": torch.ones(1, 8) * 123}, [1])
        reset = policy.evaluate_actions(
            items.get("obs"), items.get("action"), poisoned, items.get("episode_start")
        )
        self.assertTrue(torch.allclose(reset.log_prob, evaluation.log_prob))

    def test_bootstrap_uses_rnn_state_out(self):
        policy = RecurrentCategoricalPolicy(GRUActorCritic(2, 2, 8))
        data = collector(policy).collect(2).trajectories[0].data
        processed = GAEProcessor().process(data, policy)
        next_value = policy.values(data.get(("next", "obs")), data.get("state_out"))[-1]
        expected = (
            data.get("reward")[-1]
            + 0.99 * next_value
            - data.get(("policy", "extras", "value"))[-1]
        )
        self.assertTrue(torch.allclose(processed.get("advantage")[-1], expected))

    def test_ppo_objective_matches_scalar_reference_and_gradients(self):
        p = PPOPolicy(MLPActorCritic(2, 2, 8))
        data = GAEProcessor().process(
            collector(p).collect_episode().trajectories[0].data, p
        )
        ev = p.evaluate_actions(data.get("obs"), data.get("action"))
        mask = torch.ones(5, dtype=torch.bool)
        loss, metrics = ppo_objective(
            ev, data, mask, PPOConfig(normalize_advantage=False)
        )
        reference = (
            -(ev.log_prob - data.get(("policy", "extras", "logp")))
            .exp()
            .mul(data.get("advantage"))
            .mean()
            + 0.5 * (ev.value - data.get("return")).square().mean()
            - 0.01 * ev.entropy.mean()
        )
        self.assertAlmostEqual(float(metrics["ratio"]), 1.0, places=6)
        left = torch.autograd.grad(loss, tuple(p.parameters()), retain_graph=True)
        right = torch.autograd.grad(reference, tuple(p.parameters()))
        self.assertTrue(torch.allclose(loss, reference))
        for a, b in zip(left, right):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_padding_and_burn_in_do_not_enter_loss(self):
        p = RecurrentCategoricalPolicy(GRUActorCritic(2, 2, 8))
        data = GAEProcessor().process(
            collector(p).collect_episode().trajectories[0].data, p
        )
        seq = SequenceBatcher(unroll_length=3, burn_in=2).batch([data])
        self.assertEqual(int(seq.valid_mask.sum()), 5)
        self.assertEqual(len(seq.prefixes[1]), 2)
        states = seq.states(p)
        self.assertFalse(states.get("hidden").requires_grad)
        original = p.evaluate_actions(
            seq.data.get("obs"),
            seq.data.get("action"),
            states,
            seq.data.get("episode_start"),
        )
        old = seq.data.get(("policy", "extras", "logp"))
        self.assertTrue(
            torch.allclose(
                original.log_prob[seq.valid_mask], old[seq.valid_mask], atol=1e-6
            )
        )
        loss, _ = ppo_objective(original, seq.data, seq.valid_mask, PPOConfig())
        grads = torch.autograd.grad(loss, tuple(p.parameters()))
        seq.data.get(("obs", "obs"))[~seq.valid_mask] = 999
        seq.data.get("return")[~seq.valid_mask] = 999
        changed = p.evaluate_actions(
            seq.data.get("obs"),
            seq.data.get("action"),
            seq.states(p),
            seq.data.get("episode_start"),
        )
        new_loss, _ = ppo_objective(changed, seq.data, seq.valid_mask, PPOConfig())
        new_grads = torch.autograd.grad(new_loss, tuple(p.parameters()))
        self.assertTrue(torch.allclose(loss, new_loss))
        for a, b in zip(grads, new_grads):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))
        metrics = PPOLearner(p, PPOConfig(epochs=1, minibatch_size=4)).update(seq)
        self.assertEqual(metrics["gradient_updates"], 2)

    def test_terminal_truncated_and_fragment_bootstrap(self):
        p = PPOPolicy(MLPActorCritic(2, 2, 8))
        for parameter in p.parameters():
            parameter.data.zero_()
        p.module.critic.bias.data.fill_(2)
        data = collector(p, gamma=0.5, gae_lambda=0.8).collect(1).trajectories[0].data
        data.set("reward", torch.tensor([1.0]))
        for terminal, truncated, expected in (
            (True, False, -1.0),
            (False, True, 0.0),
            (False, False, 0.0),
        ):
            row = data.clone()
            row.set("terminated", torch.tensor([terminal]))
            row.set("truncated", torch.tensor([truncated]))
            row.set("discount", torch.tensor([0.0 if terminal else 0.5]))
            row.set(
                "trace_discount", torch.tensor([0.0 if terminal or truncated else 0.4])
            )
            self.assertAlmostEqual(
                float(GAEProcessor().process(row, p).get("advantage")[0]), expected
            )

    def test_dead_agent_is_not_acted_or_promoted_to_global_terminal(self):
        class DeathEnv(CounterEnv):
            def reset(self, config=None):
                self.index = 0
                return MultiEnvStep(
                    {a: AgentStep(self._observation()) for a in ("a", "b")}, False
                )

            def step(self, actions):
                self.index += 1
                if self.index == 2:
                    if set(actions) != {"b"}:
                        raise AssertionError("dead agent acted")
                return MultiEnvStep(
                    {
                        "a": AgentStep(self._observation(), 1, True),
                        "b": AgentStep(self._observation(), -1, self.index == 2),
                    },
                    self.index == 2,
                )

        p = PPOPolicy(MLPActorCritic(2, 2, 8))
        c = SynchronousStepCollector(
            DeathEnv(),
            StaticPolicyRouter({a: PolicyRef("shared", p) for a in ("a", "b")}),
        )
        result = c.collect_episode()
        by_agent = {t.key.agent_id: t.data for t in result.trajectories}
        self.assertFalse(bool(by_agent["b"].get("terminated")[0]))
        self.assertEqual(len(by_agent["a"]), 1)
        self.assertEqual(result.agent_steps, 3)

    def test_sparse_clocks_rewards_and_missing_truncation_observation(self):
        p = PPOPolicy(MLPActorCritic(2, 2, 8))
        for clock, reward in (("decision", 2.0), ("env_step", 1.5)):
            c = SparseDecisionCollector(
                AlternatingEnv(),
                StaticPolicyRouter({a: PolicyRef("shared", p) for a in ("a", "b")}),
                lambda a, s: 1.0,
                gamma=0.5,
                gae_lambda=0.8,
                discount_clock=clock,
            )
            result = c.collect_episode()
            a = next(t.data for t in result.trajectories if t.key.agent_id == "a")
            self.assertAlmostEqual(float(a.get("reward")[0]), reward)
            expected = 0.5 if clock == "decision" else 0.25
            self.assertAlmostEqual(float(a.get("discount")[0]), expected)
            self.assertAlmostEqual(float(a.get("trace_discount")[0]), expected * 0.8)

        class Missing(AlternatingEnv):
            def step(self, actions):
                return MultiEnvStep(
                    {}, True, TensorDict({"truncated": torch.tensor(True)}, [])
                )

        with self.assertRaisesRegex(ValueError, "true final"):
            SparseDecisionCollector(
                Missing(), StaticPolicyRouter({"a": PolicyRef("p", p)})
            ).collect_episode()

    def test_sparse_pending_survives_cut_and_rejects_changed_weights(self):
        p = PPOPolicy(MLPActorCritic(2, 2, 8))
        c = SparseDecisionCollector(
            AlternatingEnv(),
            StaticPolicyRouter({a: PolicyRef("shared", p) for a in ("a", "b")}),
        )
        self.assertEqual(len(c.collect(1).trajectories), 0)
        self.assertTrue(c.has_pending)
        with torch.no_grad():
            next(p.parameters()).add_(0.1)
        with self.assertRaisesRegex(RuntimeError, "weights changed"):
            c.collect(1)

    def test_shared_workflow_one_optimizer_and_parameter_conflict(self):
        p = PPOPolicy(MLPActorCritic(2, 2, 8))
        for parameter in p.parameters():
            parameter.data.zero_()

        class Opposite(CounterEnv):
            def reset(self, config=None):
                self.index = 0
                return MultiEnvStep(
                    {a: AgentStep(self._observation()) for a in ("a", "b")}, False
                )

            def step(self, actions):
                self.index += 1
                return MultiEnvStep(
                    {
                        a: AgentStep(self._observation(), r, self.index == 3)
                        for a, r in (("a", 1.0), ("b", -1.0))
                    },
                    self.index == 3,
                )

        c = SynchronousStepCollector(
            Opposite(),
            StaticPolicyRouter({a: PolicyRef("shared", p) for a in ("a", "b")}),
            gamma=1,
            gae_lambda=1,
        )
        learner = PPOLearner(p, PPOConfig(epochs=1, minibatch_size=64))
        stack = OnPolicyStack(p, GAEProcessor(1, 1), learner)
        workflow = MultiPolicyOnPolicyWorkflow(c, {"shared": stack})
        result = c.collect_episode()
        for t in result.trajectories:
            adv = stack.processor.process(t.data, p).get("advantage")
            expected = torch.tensor([3.0, 2.0, 1.0]) * (
                1 if t.key.agent_id == "a" else -1
            )
            self.assertTrue(torch.equal(adv, expected))
        metrics = workflow.step()
        self.assertEqual(metrics["gradient_updates"], 1)
        self.assertEqual(metrics["env_steps"], 3)
        self.assertEqual(metrics["agent_steps"], 6)
        self.assertEqual(workflow.weight_store.latest("shared").version, 2)
        conflict = SynchronousStepCollector(
            Opposite(),
            StaticPolicyRouter({"a": PolicyRef("a", p), "b": PolicyRef("b", p)}),
            gamma=1,
            gae_lambda=1,
        )
        with self.assertRaisesRegex(ValueError, "share parameters"):
            MultiPolicyOnPolicyWorkflow(conflict, {"a": stack, "b": stack})

    def test_router_rejects_two_instances_under_one_policy_id(self):
        left = PPOPolicy(MLPActorCritic(2, 2))
        right = PPOPolicy(MLPActorCritic(2, 2))
        with self.assertRaisesRegex(ValueError, "one policy id"):
            StaticPolicyRouter(
                {"a": PolicyRef("shared", left), "b": PolicyRef("shared", right)}
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cpu_cuda_loss_and_gradient_agree(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)
        cpu = PPOPolicy(MLPActorCritic(2, 2, 8))
        gpu = copy.deepcopy(cpu).cuda()
        data = GAEProcessor().process(
            collector(cpu).collect_episode().trajectories[0].data, cpu
        )
        results = []
        for p, device in ((cpu, "cpu"), (gpu, "cuda")):
            batch = data.to(device)
            ev = p.evaluate_actions(batch.get("obs"), batch.get("action"))
            loss, _ = ppo_objective(
                ev, batch, torch.ones(5, dtype=torch.bool, device=device), PPOConfig()
            )
            results.append(
                (
                    loss.detach().cpu(),
                    [
                        v.detach().cpu()
                        for v in torch.autograd.grad(loss, tuple(p.parameters()))
                    ],
                )
            )
        self.assertTrue(torch.allclose(results[0][0], results[1][0], atol=1e-5))
        for a, b in zip(results[0][1], results[1][1]):
            self.assertTrue(torch.allclose(a, b, atol=1e-5))
