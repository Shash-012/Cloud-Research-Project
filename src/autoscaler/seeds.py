"""Seeding helpers. Library code never touches the global numpy RNG."""
import random

import numpy as np
import torch

# Independent streams of one evaluation episode (Step 1.2).
STREAM_WORKLOAD, STREAM_SYSTEM, STREAM_CONTROLLER = 0, 1, 2


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(seed)


def child_rngs(parent: int, n: int) -> list[np.random.Generator]:
    """n independent generators derived from one parent seed."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(parent).spawn(n)]


def episode_rngs(seed: int) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """(workload, system noise, controller) streams for one episode."""
    wl, sysn, ctrl = child_rngs(seed, 3)
    return wl, sysn, ctrl
