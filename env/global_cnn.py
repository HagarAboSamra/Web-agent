"""
global_cnn.py
Encodes a full screenshot into a 256-dim global feature vector.

Deep Learning Architecture
--------------------------
  Conv1  3→16 ch   5×5 kernel  stride 2  ReLU
  Conv2  16→24 ch  5×5 kernel  stride 2  ReLU
  Conv3  24→32 ch  5×5 kernel  stride 2  ReLU
  Conv4  32→48 ch  5×5 kernel  stride 2  ReLU
  Conv5  48→32 ch  5×5 kernel  stride 2  ReLU
  FC     flat→256  ReLU

Why CNN?
  Spatial feature extraction: nearby pixels share filters (translation invariance).
  Each Conv layer learns progressively abstract features:
    Layer 1 → edges and colours
    Layer 2 → textures and gradients
    Layer 3 → shapes and patterns
    Layer 4 → semantic regions
    Layer 5 → high-level visual concepts

He Initialisation:  std = sqrt(2 / fan_in)
  Prevents vanishing/exploding gradients in deep ReLU networks.
"""

import numpy as np

VIEWPORTS = {"desktop": (1280, 720), "tablet": (768, 1024), "mobile": (375, 667), "miniwob": (160, 210)}
FILTERS   = [16, 24, 32, 48, 32]
KERNEL    = 5
STRIDE    = 2


class _Conv:
    """Single 2-D conv layer: 5×5 kernel, stride 2, ReLU."""

    def __init__(self, in_ch, out_ch, seed=0):
        rng      = np.random.default_rng(seed)
        std      = np.sqrt(2.0 / (in_ch * KERNEL * KERNEL))   # He init
        self.W   = rng.normal(0, std, (out_ch, in_ch, KERNEL, KERNEL)).astype(np.float32)
        self.b   = np.zeros(out_ch, dtype=np.float32)

    def forward(self, x):
        # x: (H, W, in_ch)
        H, W, _ = x.shape
        Ho = (H - KERNEL) // STRIDE + 1
        Wo = (W - KERNEL) // STRIDE + 1
        out = np.zeros((Ho, Wo, len(self.b)), dtype=np.float32)
        for i in range(Ho):
            for j in range(Wo):
                patch = x[i*STRIDE:i*STRIDE+KERNEL, j*STRIDE:j*STRIDE+KERNEL, :]
                for f in range(len(self.b)):
                    out[i, j, f] = np.sum(patch * self.W[f].transpose(1, 2, 0)) + self.b[f]
        return np.maximum(out, 0.0)   # ReLU


class _FC:
    """Fully connected layer with optional ReLU."""

    def __init__(self, in_dim, out_dim, relu=True, seed=0):
        rng      = np.random.default_rng(seed)
        self.W   = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, out_dim)).astype(np.float32)
        self.b   = np.zeros(out_dim, dtype=np.float32)
        self.relu = relu

    def forward(self, x):
        out = x @ self.W + self.b
        return np.maximum(out, 0.0) if self.relu else out


class GlobalCNN:
    """
    Screenshot → 256-dim global feature vector.

    Parameters
    ----------
    viewport : str | tuple[int, int]
    fc_out   : int   output vector length (default 256)
    seed     : int
    """

    def __init__(self, viewport="desktop", fc_out=256, seed=42):
        w, h    = VIEWPORTS[viewport.lower()] if isinstance(viewport, str) else viewport
        in_ch   = 3
        self.convs = []
        for i, out_ch in enumerate(FILTERS):
            self.convs.append(_Conv(in_ch, out_ch, seed=seed + i))
            in_ch = out_ch
        # compute spatial size after all conv layers
        for _ in FILTERS:
            w = (w - KERNEL) // STRIDE + 1
            h = (h - KERNEL) // STRIDE + 1
        self.fc     = _FC(w * h * FILTERS[-1], fc_out, seed=seed + 10)
        self.fc_out = fc_out

    def forward(self, x):
        """
        x : (W, H, 3) float32 from ScreenProcessor.process()

        Returns
        -------
        global_vec   : (fc_out,) float32
        feature_maps : list[np.ndarray]  — intermediate activations for LocalCNN
        """
        feat = x.transpose(1, 0, 2)   # (W,H,3) → (H,W,3)
        maps = []
        for conv in self.convs:
            feat = conv.forward(feat)
            maps.append(feat)
        return self.fc.forward(feat.ravel()), maps
