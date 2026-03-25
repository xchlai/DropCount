from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import ExperimentConfig, load_config, save_config
from models import VolumeAwareSetTransformerRegressor
from simulator import DropletDigitalPCRSimulator, FixedSimulationDataset, OnlineSimulationDataset, collate_samples
from utils import detect_device, ensure_dir, plot_training_curves, save_json, set_global_seed


def build_model(config: ExperimentConfig) -> VolumeAwareSetTransformerRegressor:
    return VolumeAwareSetTransformerRegressor(
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        num_latents=config.model.num_latents,
        num_heads=config.model.num_heads,
        num_self_attn_layers=config.model.num_self_attn_layers,
        dropout=config.model.dropout,
        fourier_features=config.model.fourier_features,
        use_rmsnorm=config.model.use_rmsnorm,
    )


def compute_loss(outputs: Dict[str, torch.Tensor], targets: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    target_log = torch.log1p(targets)
    pred_log = outputs["pred_log_copies"]
    if config.training.loss_name == "mse_log":
        loss_log = nn.functional.mse_loss(pred_log, target_log)
    elif config.training.loss_name == "huber_log":
        loss_log = nn.functional.huber_loss(pred_log, target_log, delta=config.training.huber_delta)
    else:
        raise ValueError(f"Unsupported loss: {config.training.loss_name}")
    pred_linear = outputs["pred_copies"]
    loss_linear = nn.functional.l1_loss(pred_linear, targets) / (targets.mean().clamp_min(1.0))
    return loss_log + config.training.linear_loss_weight * loss_linear


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, config: ExperimentConfig) -> float:
    model.eval()
    losses: List[float] = []
    for batch in loader:
        outputs = model(
            batch["volume_fractions"].to(device),
            batch["labels"].to(device),
            batch["false_positive_rate"].to(device),
            batch["mask"].to(device),
        )
        loss = compute_loss(outputs, batch["true_total_copies"].to(device), config)
        losses.append(float(loss.item()))
    return float(sum(losses) / max(len(losses), 1))


def train(config: ExperimentConfig) -> Path:
    set_global_seed(config.simulation.random_seed)
    run_dir = ensure_dir(config.run_dir)
    device = detect_device(config.training.device)
    simulator = DropletDigitalPCRSimulator(config)

    train_dataset = OnlineSimulationDataset(
        simulator=simulator,
        num_samples=config.simulation.train_samples_per_epoch,
        distributions=config.simulation.distributions.train_names,
        cv_values=config.simulation.distributions.cv_values,
        n_droplets_choices=[config.simulation.n_droplets],
        seed_offset=0,
    )
    val_dataset = FixedSimulationDataset(
        simulator=simulator,
        num_samples=config.simulation.val_samples,
        distributions=config.simulation.distributions.train_names,
        cv_values=config.simulation.distributions.cv_values,
        n_droplets_choices=[config.simulation.n_droplets],
        seed_offset=999,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        collate_fn=collate_samples,
        num_workers=config.training.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, collate_fn=collate_samples)

    model = build_model(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(config.training.epochs, 1),
        eta_min=config.training.min_learning_rate,
    )
    scaler = torch.amp.GradScaler(enabled=config.training.amp and device.type == "cuda")

    best_val = float("inf")
    best_epoch = -1
    history: List[Dict[str, float]] = []
    checkpoint_path = run_dir / "best_model.pt"

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        train_losses: List[float] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config.training.amp and device.type == "cuda"):
                outputs = model(
                    batch["volume_fractions"].to(device),
                    batch["labels"].to(device),
                    batch["false_positive_rate"].to(device),
                    batch["mask"].to(device),
                )
                loss = compute_loss(outputs, batch["true_total_copies"].to(device), config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))
        scheduler.step()
        val_loss = evaluate_loss(model, val_loader, device, config)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(sum(train_losses) / max(len(train_losses), 1)),
            "val_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "best_val": best_val,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        if epoch - best_epoch >= config.training.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    plot_training_curves(history, run_dir / "training_curves.png")
    save_config(config, run_dir / "config.yaml")
    save_json({"best_val": best_val, "best_epoch": best_epoch, "device": str(device)}, run_dir / "summary.json")
    if config.output.save_validation_dataset:
        torch.save(val_dataset.samples, run_dir / "validation_dataset.pt")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the deep-learning ddPCR quantification model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-droplets", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: Dict[str, Dict[str, object]] = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.n_droplets is not None:
        overrides.setdefault("simulation", {})["n_droplets"] = args.n_droplets
    if args.run_name is not None:
        overrides.setdefault("output", {})["run_name"] = args.run_name
    config = load_config(args.config, overrides=overrides or None)
    checkpoint = train(config)
    print(f"Saved best checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
