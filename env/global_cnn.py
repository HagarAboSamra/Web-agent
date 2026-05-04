"""
global_cnn.py

Processes the full normalised screenshot through a deep convolutional
network to produce a holistic global visual feature vector.

Architecture summary
--------------------
5 convolutional layers with 5×5 filters, stride 2.
Filter depths: [16, 24, 32, 48, 32].
Followed by an optional extra feed-forward layer (activated for larger
viewports) and a final FC projection to ``fc_out`` dimensions.

The class is framework-agnostic: weights are plain NumPy arrays.
Intermediate feature maps are returned so ``LocalCNN`` can attend to them.

Viewport configuration
-----------------------
viewport : str | tuple[int, int]
    Matches the preset used in ``ScreenProcessor``.
    "desktop" (1280×720), "tablet" (768×1024), "mobile" (375×667),
    or a custom (W, H) tuple.
    Larger viewports automatically enable the extra feed-forward layer.
"""

import numpy as np
from typing import List, Tuple


VIEWPORT_PRESETS: dict[str, Tuple[int, int]] = {
    "desktop": (1280, 720),
    "tablet":  (768,  1024),
    "mobile":  (375,  667),
}

# Viewports with area > this threshold get the extra FF layer
_EXTRA_FF_AREA_THRESHOLD = 375 * 667   # ~250 k px²


class ConvLayer:
    """
    2-D convolutional layer (NumPy reference implementation).
    5×5 filters, stride 2, valid padding, ReLU activation.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 5,
        stride:       int = 2,
        seed:         int = 0,
    ) -> None:
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        rng    = np.random.default_rng(seed)
        fan_in = in_channels * kernel_size * kernel_size
        std    = np.sqrt(2.0 / fan_in)
        self.weights = rng.normal(
            0, std,
            (out_channels, in_channels, kernel_size, kernel_size),
        ).astype(np.float32)
        self.bias = np.zeros(out_channels, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (H, W, C_in)  →  (H_out, W_out, C_out), ReLU applied."""
        H, W, _ = x.shape
        k, s    = self.kernel_size, self.stride
        H_out   = (H - k) // s + 1
        W_out   = (W - k) // s + 1
        out     = np.zeros((H_out, W_out, self.out_channels), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                patch = x[i*s:i*s+k, j*s:j*s+k, :]
                for f in range(self.out_channels):
                    out[i, j, f] = (
                        np.sum(patch * self.weights[f].transpose(1, 2, 0))
                        + self.bias[f]
                    )
        return np.maximum(out, 0.0)


class FullyConnected:
    """Single linear layer + optional ReLU."""

    def __init__(self, in_dim: int, out_dim: int, activation: bool = True, seed: int = 0) -> None:
        rng      = np.random.default_rng(seed)
        std      = np.sqrt(2.0 / in_dim)
        self.W   = rng.normal(0, std, (in_dim, out_dim)).astype(np.float32)
        self.b   = np.zeros(out_dim, dtype=np.float32)
        self.act = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.W + self.b
        return np.maximum(out, 0.0) if self.act else out


class GlobalCNN:
    """
    Holistic screen encoder.

    Encodes a full normalised screenshot into a compact global feature
    vector and also exposes every intermediate convolutional feature map
    for use by ``LocalCNN``.

    Parameters
    ----------
    viewport : str | tuple[int, int]
        Viewport preset or explicit (W, H).  Must match ``ScreenProcessor``.
    fc_out   : int
        Dimension of the final global feature vector (default 256).
    seed     : int
        Random seed for reproducible weight initialisation.
    """

    _FILTERS     = [16, 24, 32, 48, 32]
    _KERNEL_SIZE = 5
    _STRIDE      = 2

    def __init__(
        self,
        viewport: str | Tuple[int, int] = "desktop",
        fc_out:   int = 256,
        seed:     int = 42,
    ) -> None:
        if isinstance(viewport, str):
            key = viewport.lower()
            if key not in VIEWPORT_PRESETS:
                raise ValueError(f"Unknown viewport preset '{viewport}'.")
            self.input_w, self.input_h = VIEWPORT_PRESETS[key]
        else:
            self.input_w, self.input_h = int(viewport[0]), int(viewport[1])

        self.fc_out = fc_out
        area = self.input_w * self.input_h

        # Build conv stack
        in_ch = 3
        self.conv_layers: List[ConvLayer] = []
        for i, out_ch in enumerate(self._FILTERS):
            self.conv_layers.append(
                ConvLayer(in_ch, out_ch, self._KERNEL_SIZE, self._STRIDE, seed + i)
            )
            in_ch = out_ch

        # Compute flattened spatial dim after all conv layers
        w, h = self.input_w, self.input_h
        for _ in self._FILTERS:
            w = (w - self._KERNEL_SIZE) // self._STRIDE + 1
            h = (h - self._KERNEL_SIZE) // self._STRIDE + 1
        flat_dim = w * h * self._FILTERS[-1]

        # Extra FF layer for larger viewports
        self.extra_ff: FullyConnected | None = None
        if area > _EXTRA_FF_AREA_THRESHOLD:
            self.extra_ff = FullyConnected(flat_dim, flat_dim, activation=True, seed=seed + 10)

        self.fc = FullyConnected(flat_dim, fc_out, activation=True, seed=seed + 11)

    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Parameters
        ----------
        x : np.ndarray
            Normalised tensor of shape ``(W, H, 3)`` from ScreenProcessor.

        Returns
        -------
        global_features : np.ndarray
            1-D float32 vector, length ``fc_out``.
        intermediate_maps : list[np.ndarray]
            Per-layer feature maps ``(H_l, W_l, C_l)`` for LocalCNN.
        """
        feat  = x.transpose(1, 0, 2)   # (W, H, C) → (H, W, C)
        imaps: List[np.ndarray] = []

        for layer in self.conv_layers:
            feat = layer.forward(feat)
            imaps.append(feat)

        flat = feat.ravel()
        if self.extra_ff is not None:
            flat = self.extra_ff.forward(flat)

        return self.fc.forward(flat), imaps

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        extra = "+FF" if self.extra_ff else ""
        return (
            f"GlobalCNN("
            f"viewport={self.input_w}×{self.input_h}, "
            f"layers={len(self.conv_layers)}{extra}, "
            f"filters={self._FILTERS}, "
            f"fc_out={self.fc_out})"
        )
