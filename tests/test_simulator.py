import numpy as np

from config import load_config
from simulator import DropletDigitalPCRSimulator, OnlineSimulationDataset


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


def test_false_positive_rate_applies_when_no_true_copies():
    config = load_config(
        "configs/default.yaml",
        overrides={"simulation": {"n_droplets": 500, "false_positive_rate_range": [0.3, 0.3]}},
    )
    simulator = DropletDigitalPCRSimulator(config)
    sample = simulator.simulate_sample(n_true=0, distribution_name="monodisperse", cv=0.0)
    assert np.isclose(sample.false_positive_rate, 0.3)
    assert np.isclose(sample.metadata["false_positive_rate"], 0.3)
    assert sample.labels.mean() > 0.15


def test_online_dataset_reshuffles_each_epoch():
    simulator = build_simulator()
    dataset = OnlineSimulationDataset(simulator=simulator, num_samples=5)
    first_epoch = list(dataset)
    second_epoch = list(dataset)
    first_values = np.array([float(x["true_total_copies"]) for x in first_epoch])
    second_values = np.array([float(x["true_total_copies"]) for x in second_epoch])
    assert not np.array_equal(first_values, second_values)
