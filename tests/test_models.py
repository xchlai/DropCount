import torch

from models import VolumeAwareSetTransformerRegressor


def test_model_forward_shape_and_non_negative_output():
    model = VolumeAwareSetTransformerRegressor(hidden_dim=32, latent_dim=32, num_latents=8, num_self_attn_layers=1)
    volume_fractions = torch.rand(4, 20)
    volume_fractions = volume_fractions / volume_fractions.sum(dim=1, keepdim=True)
    labels = torch.randint(0, 2, (4, 20)).float()
    mask = torch.ones(4, 20, dtype=torch.bool)
    outputs = model(volume_fractions, labels, mask)
    assert outputs["pred_copies"].shape == (4,)
    assert outputs["pred_log_copies"].shape == (4,)
    assert torch.all(outputs["pred_copies"] >= 0)
