import numpy as np

from dht.toolcall.depth_policy import (
    DEPTHS,
    DepthPolicy,
    context_features,
    loop_stats,
    reward,
)


def test_reward_cost_term():
    assert reward(True, 1.0, 0.3) == 1.0
    assert abs(reward(True, 4.0, 0.3) - 0.7) < 1e-12
    assert abs(reward(False, 2.5, 0.6) - (-0.3)) < 1e-12


def test_probs_sum_to_one_and_roundtrip():
    pol = DepthPolicy(5, seed=1)
    s = np.arange(5.0)
    p = pol.probs(s)
    assert p.shape == (4,) and abs(p.sum() - 1) < 1e-12
    again = DepthPolicy.from_dict(pol.to_dict())
    assert np.allclose(again.probs(s), p)


def test_reinforce_learns_feature_dependent_depth():
    """Toy chain: easy turns (feature 0) are solved at depth 1, hard turns (feature 1) need 4.

    With a cost on loops, the policy must learn shallow-on-easy, deep-on-hard."""
    rng = np.random.default_rng(0)
    pol = DepthPolicy(2, seed=0)
    for _ in range(400):
        batch = []
        for _ in range(16):
            states, depths, ok = [], [], True
            for _t in range(3):
                hard = rng.random() < 0.5
                s = np.array([1.0, 0.0]) if not hard else np.array([0.0, 1.0])
                pol.norm.update(s)
                d = pol.act(s, rng)
                states.append(s)
                depths.append(d)
                ok &= d >= (4 if hard else 1)
            loops = float(np.mean(depths))
            batch.append((states, depths, reward(ok, loops, lam=0.6)))
        pol.update(batch, lr=0.5, beta=0.01)
    assert pol.act(np.array([1.0, 0.0])) == 1
    assert pol.act(np.array([0.0, 1.0])) == 4


def test_features_shapes():
    logits = [np.random.default_rng(i).normal(size=50) for i in range(4)]
    f = loop_stats(logits, [0.1, 0.2, 0.3, 0.4])
    assert f.shape == (16,)
    assert f[15] == 1.0  # loop 4 agrees with itself
    c = context_features(turn=7, n_calls=3, last_error=True, prompt_tokens=1000, n_tools=5)
    assert c.shape == (10,) and c[5] == 1.0 and c[7] == 1.0
    assert len(DEPTHS) == 4
