import numpy as np
from autoscale.models.rt_nn import RTModel, NNConfig, layer_sizes


def test_layer_sizes_follow_paper_formula():
    assert layer_sizes(10) == (21, 12)                                         # H1=2n+1, H2=ceil(H1/2)+1


def test_nn_learns_hockey_stick():
    rng = np.random.default_rng(0)
    u = rng.uniform(0.1, 0.95, 600); y = 100 / (1 - u)
    m = RTModel(1, NNConfig(max_epochs=300, patience=40)).fit(u[:, None], y, seed=0)
    p = m.predict(np.array([[0.3], [0.9]]))
    assert abs(p[0] - 143) < 40 and p[1] > 600
    mu, sd = m.predict_dist(np.array([[0.5]])); assert (sd == 0).all()


def test_seed_sets_are_disjoint(cfg):
    assert cfg["tuning_seed"] not in cfg["eval_seeds"] and cfg["train_seed"] not in cfg["eval_seeds"]
    assert cfg["tuning_seed"] != cfg["train_seed"]
