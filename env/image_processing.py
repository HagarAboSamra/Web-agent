"""
image_processing.py

Imports WebScreenshot from screenshot.py, captures the page in memory,
and feeds the 210 × 160 × 3 RGB array directly into Global_CNN —
no intermediate files involved.

Architecture:
  Input  : (B, 3, 210, 160)   – CHW after permute
  Conv1  : 5×5, stride 2, 16 filters   → (B, 16, 103, 78)
  Conv2  : 5×5, stride 2, 24 filters   → (B, 24,  50, 37)
  Conv3  : 5×5, stride 2, 32 filters   → (B, 32,  23, 17)
  Conv4  : 5×5, stride 2, 48 filters   → (B, 48,  10,  7)
  Conv5  : 5×5, stride 2, 32 filters   → (B, 32,   3,  2)
  AvgPool: global average pool          → (B, 32)
  FC1    : Linear(32, 384)  + ReLU
  FC2    : Linear(384, num_logits)      → logits

Dependencies:
    pip install torch numpy selenium pillow chromedriver-autoinstaller
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from screenshot import WebScreenshot, TARGET_SHAPE


# ── Tensor conversion ──────────────────────────────────────────────────────────

def array_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """
    Convert a uint8 (H, W, C) NumPy array to a normalised float32
    CHW tensor ready for Global_CNN.

    Normalisation : pixels / 255  →  [0, 1]
    Output shape  : (1, 3, 210, 160)
    """
    tensor = torch.from_numpy(rgb).float() / 255.0   # (H, W, C)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)    # (1, C, H, W)
    return tensor


def screenshot_to_tensor(url: str, **kwargs) -> tuple[np.ndarray, torch.Tensor]:
    """
    Convenience function: capture a URL and return both the raw array
    and the ready-to-use tensor in one call.

    Parameters
    ----------
    url : str
    **kwargs : forwarded to WebScreenshot (wait_seconds, viewport_width, …)

    Returns
    -------
    rgb_array : np.ndarray  shape (210, 160, 3) uint8
    tensor    : torch.Tensor  shape (1, 3, 210, 160) float32 in [0,1]
    """
    ws        = WebScreenshot(url, **kwargs)
    rgb_array = ws.capture()
    return rgb_array, array_to_tensor(rgb_array)


# ── Global CNN ─────────────────────────────────────────────────────────────────

class Global_CNN(nn.Module):
    """
    Six-layer feedforward network for global screenshot understanding.

    Parameters
    ----------
    num_logits : int
        Size of the final output (number of action/class logits).
    in_channels : int
        Number of input image channels. Default 3 (RGB).

    Usage
    -----
    >>> rgb, tensor = screenshot_to_tensor("https://example.com")
    >>> model  = Global_CNN()
    >>> logits = model(tensor)                  # (1, 128)
    >>> feats  = model.extract_features(tensor) # (1, 384)
    """

    _CONV_CHANNELS = [16, 24, 32, 48, 32]

    def __init__(self, num_logits: int = 128, in_channels: int = 3):
        super().__init__()
        self.num_logits = num_logits

        # ── 5 Convolutional layers ────────────────────────────────────────────
        conv_layers: list[nn.Module] = []
        prev_ch = in_channels
        for out_ch in self._CONV_CHANNELS:
            conv_layers.append(
                nn.Conv2d(
                    in_channels=prev_ch,
                    out_channels=out_ch,
                    kernel_size=5,
                    stride=2,
                    padding=0,
                    bias=True,
                )
            )
            conv_layers.append(nn.ReLU(inplace=True))
            prev_ch = out_ch

        self.conv_backbone   = nn.Sequential(*conv_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1             = nn.Linear(self._CONV_CHANNELS[-1], 384)
        self.fc2             = nn.Linear(384, num_logits)

        self._init_weights()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 210, 160)  float32 in [0, 1]

        Returns
        -------
        logits : (B, num_logits)
        """
        x      = self.conv_backbone(x)           # (B, 32, H', W')
        x      = self.global_avg_pool(x)         # (B, 32, 1, 1)
        x      = x.flatten(start_dim=1)          # (B, 32)
        x      = F.relu(self.fc1(x))             # (B, 384)
        return   self.fc2(x)                     # (B, num_logits)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """384-dim representation after FC1 (before logit projection)."""
        x = self.conv_backbone(x)
        x = self.global_avg_pool(x).flatten(start_dim=1)
        return F.relu(self.fc1(x))

    def conv_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Raw spatial feature map after all 5 conv layers — shape (B, 32, H', W')."""
        return self.conv_backbone(x)

    # ── Weight init ───────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Factory: build directly from a URL ────────────────────────────────────

    @classmethod
    def from_url(
        cls,
        url: str,
        num_logits: int = 128,
        **screenshot_kwargs,
    ) -> tuple["Global_CNN", torch.Tensor, torch.Tensor]:
        """
        Capture a screenshot, build the model, and run a forward pass.

        Returns
        -------
        model   : Global_CNN
        tensor  : preprocessed input  (1, 3, 210, 160)
        logits  : model output        (1, num_logits)
        """
        _, tensor = screenshot_to_tensor(url, **screenshot_kwargs)
        model     = cls(num_logits=num_logits)
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
        return model, tensor, logits


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"

    print(f"[image_processing] Capturing {url} …")
    model, tensor, logits = Global_CNN.from_url(url)

    print(f"Input tensor shape  : {tuple(tensor.shape)}")
    print(f"Logits shape        : {tuple(logits.shape)}")
    print(f"Logits (first 8)    : {logits[0, :8].tolist()}")
    print(f"Feature map shape   : {tuple(model.conv_feature_map(tensor).shape)}")
    print(f"FC1 features shape  : {tuple(model.extract_features(tensor).shape)}")
