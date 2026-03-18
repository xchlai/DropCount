# Deep Learning for Polydisperse Droplet Digital PCR Quantification

This project implements a complete, runnable Python research codebase for studying how droplet polydispersity affects digital PCR quantification. It compares three methods:

1. **Deep learning**: a permutation-invariant Transformer-style set regressor.
2. **Naive equal-volume Poisson estimator**: assumes every droplet has the same volume.
3. **Volume-aware maximum likelihood estimator (MLE)**: uses the true droplet volume fractions.

The code is designed for Monte Carlo simulation, training, evaluation, figure generation, and reproducible testing.

---

## Project structure

```text
.
├── baselines.py
├── config.py
├── configs/
│   └── default.yaml
├── evaluate.py
├── models.py
├── README.md
├── requirements.txt
├── simulator.py
├── train.py
├── utils.py
└── tests/
    ├── test_baselines.py
    ├── test_models.py
    └── test_simulator.py
```

---

## Problem formulation

Each sample contains many droplets. For droplet `i`:

- Relative volume fraction: `f_i > 0`
- Binary measurement result: `y_i ∈ {0, 1}`
- The fractions satisfy `sum_i f_i = 1`

The target is the true total molecule count `N_true`. Because total reaction volume is normalized to one, total copies and concentration are numerically identical here.

### Generative model

By default, the simulator uses the **fixed-total multinomial loading model**:

- Draw or set a true integer total molecule count `N_true`
- Allocate molecules to droplets by
  `counts ~ Multinomial(N_true, f)`
- Convert counts to binary positivity:
  `y_i = 1[count_i > 0]`

An alternative **Poisson loading mode** is also implemented:

- `count_i ~ Poisson(N_true * f_i)`

---

## Statistical baselines

### 1) Naive equal-volume estimator

Assume all droplets have equal fraction `1/n`. Then

- Single droplet positive probability:
  `p = 1 - exp(-N / n)`
- If `z` droplets are negative, then
  `z / n ≈ exp(-N / n)`
- Therefore the estimator is
  `N_hat = -n * log(z / n)`

The implementation returns `0` for all-negative samples and clips all-positive samples to a configurable `max_copy_cap` while marking them as saturated.

### 2) Volume-aware MLE

Given true droplet fractions `f_i`, the positive probability for droplet `i` under total copy number `N` is

`p_i(N) = 1 - exp(-N f_i)`

For observed binary labels `y_i`, the log-likelihood is

`log L(N) = Σ_i [ y_i log(1 - exp(-N f_i)) + (1 - y_i)(-N f_i) ]`

The code uses a numerically stable implementation of `log(1 - exp(-x))` and optimizes `N >= 0` with `scipy.optimize.minimize_scalar`.

---

## Deep learning model

The deep model is `VolumeAwareSetTransformerRegressor`, a **permutation-invariant set model** with a Perceiver-style latent bottleneck.

### Input droplet token features

For each droplet it uses order-independent features such as:

- `f_i`
- `log(f_i)`
- `y_i`
- `f_i * y_i`
- `sqrt(f_i)`
- standardized `f_i`
- optional Fourier features on `f_i`

### Encoder architecture

1. Droplet features are embedded independently.
2. Learnable latent tokens cross-attend to droplet tokens.
3. Several latent self-attention blocks refine the global representation.
4. The pooled latent representation is regressed to `log1p(N_true)`.
5. Final copy-number output is `exp(pred_log) - 1`, clamped to remain non-negative.

This design is permutation-invariant and avoids full `O(n^2)` attention across all droplets.

---

## Training objective

The default loss is a hybrid objective:

- **Primary**: Huber loss on `log1p(N_true)`
- **Auxiliary**: small weighted L1 loss on the linear copy-number scale

This supports the wide dynamic range typical of digital PCR.

Additional training features:

- PyTorch 2.x
- mixed precision on CUDA
- cosine learning rate schedule
- gradient clipping
- early stopping
- model checkpointing
- deterministic seeding

---

## Simulation capabilities

Implemented droplet size distributions:

- `monodisperse`
- `lognormal`
- `gamma`
- `truncated_normal`
- `uniform`
- `two_point`
- `binomial_mapped`
- `custom_callable` extension hook

The simulator supports:

- configurable droplet count
- configurable true-copy range
- `log_uniform_integer`, `uniform_integer`, and `custom` sampling modes
- online training generation
- fixed validation and test datasets for reproducibility
- cross-distribution evaluation using unseen droplet volume distributions

---

## Metrics and plots

`evaluate.py` compares all three methods and reports:

- MAE
- RMSE
- RMSLE
- median relative error
- mean relative bias
- R²
- Pearson correlation
- Spearman correlation
- mean absolute error in log-space

It also saves grouped summary tables:

- overall
- by CV
- by distribution
- by copy-number bin
- by saturation subset

Generated figures include:

- predicted vs true (log-log scatter)
- relative error vs true copies
- method error boxplot
- performance vs CV
- MAE by distribution

---

## Expected scientific behavior

You should typically expect:

- In **monodisperse** conditions and at **low/moderate concentration**, the naive equal-volume estimator can already perform well.
- As **CV increases** and samples approach **saturation**, droplet size heterogeneity matters much more.
- The **volume-aware MLE** should outperform the naive estimator when true droplet volume fractions are available.
- The deep model aims to learn robust quantification across broad dynamic ranges and heterogeneous droplet distributions, and can be compared directly against the MLE to study whether it matches or surpasses it under complex conditions.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Minimal runnable demo

### Train

```bash
python train.py --config configs/default.yaml --run-name demo_run
```

### Evaluate

```bash
python evaluate.py --config configs/default.yaml --run-dir outputs/demo_run
```

### Run tests

```bash
pytest -q
```

---

## Command line examples

Train with a different droplet count:

```bash
python train.py --config configs/default.yaml --n-droplets 5000 --epochs 8 --run-name droplets_5000
```

Evaluate a specific checkpoint:

```bash
python evaluate.py --config configs/default.yaml --checkpoint outputs/demo_run/best_model.pt
```

---

## Output files

A training run stores artifacts in `outputs/<run_name>/`, typically including:

- `best_model.pt`
- `config.yaml`
- `summary.json`
- `training_history.csv`
- `training_curves.png`
- `validation_dataset.pt`
- `evaluation/*.csv`
- `evaluation/*.png`

---

## Notes on numerical stability

The project carefully handles several unstable expressions common in PCR likelihoods:

- `log(1 - exp(-x))` is evaluated through a stable branch implementation.
- logarithms and divisions are protected by small epsilons.
- all-positive samples are treated as saturated and clipped to a user-configurable cap.
- all-negative samples return zero for the statistical baselines.

---

## Extending the simulator

You can register a custom volume sampler by extending `VolumeSampler` in `simulator.py` and then selecting `custom_callable`.

---

## Reproducibility

All major stochastic components accept or derive from a fixed seed. Validation and test sets are generated deterministically from config-controlled seeds.
