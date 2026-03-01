"""
Codex environment test entry point.
Lightweight execution (no heavy training).
"""

import torch
from res2unet_seismic import Res2UNet

def test_environment():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Res2UNet().to(device)

    x = torch.randn(1, 1, 64, 64).to(device)
    y = model(x)

    print("✅ Codex Environment Working")
    print("Output shape:", y.shape)

if __name__ == "__main__":
    test_environment()
