"""
local_cnn.py
Cursor-conditioned local feature extractor.

A Gaussian mask is placed at the current mouse position over the deepest
CNN feature map. Weighted average pooling + FC → 128-dim vector.

Why local attention?
  GlobalCNN sees the full page.
  LocalCNN asks "what is near the cursor right now?" — critical for
  hover menus, context detection, and click decisions.
"""

import numpy as np


class LocalCNN:
    """
    Parameters
    ----------
    sigma      : float  Gaussian radius in feature-map pixels (default 2.0)
    output_dim : int    length of output vector (default 128)
    layer_idx  : int    which GlobalCNN map to use; -1 = deepest
    seed       : int
    """

    def __init__(self, sigma=2.0, output_dim=128, layer_idx=-1, seed=99):
        self.sigma      = sigma
        self.output_dim = output_dim
        self.layer_idx  = layer_idx
        self._seed      = seed
        self._W         = None   # built lazily
        self._b         = None

    def forward(self, feature_maps, mouse_x, mouse_y, screen_w, screen_h):
        """
        Parameters
        ----------
        feature_maps : list[np.ndarray]  from GlobalCNN.forward()
        mouse_x/y    : float  cursor position in viewport pixels
        screen_w/h   : int    viewport dimensions

        Returns
        -------
        np.ndarray  shape (output_dim,) float32
        """
        fmap      = feature_maps[self.layer_idx]   # (H, W, C)
        H, W, C   = fmap.shape
        fx        = mouse_x / screen_w * W
        fy        = mouse_y / screen_h * H
        mask      = self._gaussian(fx, fy, H, W)
        pooled    = (fmap * mask[:, :, None]).sum((0, 1)) / (mask.sum() + 1e-8)
        return self._fc(pooled, C)

    def _gaussian(self, cx, cy, H, W):
        ys  = np.arange(H, dtype=np.float32)
        xs  = np.arange(W, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        mask = np.exp(-((gx - cx)**2 + (gy - cy)**2) / (2 * self.sigma**2))
        return (mask / (mask.sum() + 1e-8)).astype(np.float32)

    def _fc(self, vec, in_dim):
        """Lazy FC layer — built on first call."""
        if self._W is None or self._W.shape[0] != in_dim:
            rng     = np.random.default_rng(self._seed)
            self._W = rng.normal(0, np.sqrt(2.0 / in_dim),
                                 (in_dim, self.output_dim)).astype(np.float32)
            self._b = np.zeros(self.output_dim, dtype=np.float32)
        return np.maximum(vec @ self._W + self._b, 0.0)
