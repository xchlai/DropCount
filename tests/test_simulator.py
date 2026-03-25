import numpy as np

from config import load_config
from simulator import DropletDigitalPCRSimulator


def build_simulator():
    config = load_config("configs/default.yaml", overrides={"simulation": {"n_droplets": 128}})
    return DropletDigitalPCRSimulator(config)


def test_volume_fractions_sum_to_one_and_positive():
    simulator = build_simulator()
    f = simulator.generate_volume_fractions(distribution_name="gamma", cv=0.3)
    assert np.isclose(f.sum(), 1.0, atol=1e-6)
    assert np.all(f > 0)


def test_monodisperse_fractions_equal():
    simulator = build_simulator()
    f = simulator.generate_volume_fractions(distribution_name="monodisperse", cv=0.0)
    assert np.allclose(f, np.full_like(f, 1.0 / len(f)))


def test_fixed_total_multinomial_counts_sum_to_true_total():
    simulator = build_simulator()
    sample = simulator.simulate_sample(n_true=321, distribution_name="lognormal", cv=0.2)
    assert int(sample.counts.sum()) == 321
