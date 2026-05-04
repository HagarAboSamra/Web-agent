"""
local_cnn.py

Produces a cursor-conditioned localised feature vector by soft-attending
to the region of the CNN feature map nearest the current mouse position.

The mouse location (tracked by Playwright via ``page.mouse.move``) acts as
the centre of a 2-D Gaussian mask that is broadcast over the chosen
intermediate CNN feature map.  The weighted average pool yields a compact
vector that drives fine-grained click, scroll, and keyboard action decisions.

No Playwright-specific imports are needed here — the module operates purely
on NumPy arrays supplied by ``GlobalCNN`` and ``EnvironmentHandler``.
"""

import numpy as np
from typing import List


class LocalCNN:
    """
    Cursor-conditioned local feature extractor.

    Parameters
    ----------
    attention_sigma : float
        Standard deviation (in feature-map pixels) of the Gaussian mask.
    output_dim : int
        Dimension of the output localised feature vector.
    layer_index : int
        Which intermediate map from ``GlobalCNN`` to attend over.
        ``-1`` → deepest (most abstract); ``0`` → shallowest (most spatial).
    seed : int
        Random seed for lazy FC weight initialisation.
    """

    def __init__(
        self,
        attention_sigma: float = 2.0,
        output_dim:      int   = 128,
        layer_index:     int   = -1,
        seed:            int   = 99,
    ) -> None:
        self.attention_sigma = attention_sigma
        self.output_dim      = output_dim
        self.layer_index     = layer_index
        self._seed           = seed
        self._fc_W: np.ndarray | None = None
        self._fc_b: np.ndarray | None = None

    # ------------------------------------------------------------------

    def forward(
        self,
        intermediate_maps: List[np.ndarray],
        mouse_x:           float,
        mouse_y:           float,
        screen_width:      int,
        screen_height:     int,
    ) -> np.ndarray:
        """
        Extract a cursor-conditioned local feature vector.

        Parameters
        ----------
        intermediate_maps : list[np.ndarray]
            Convolutional feature maps from ``GlobalCNN.forward``.
            Each is ``(H_l, W_l, C_l)``.
        mouse_x, mouse_y : float
            Current cursor position in Playwright screen-pixel coordinates.
            Playwright reports these in CSS pixels relative to the viewport
            origin (top-left = 0, 0).
        screen_width, screen_height : int
            Viewport dimensions used to scale mouse coords into map coords.

        Returns
        -------
        np.ndarray
            1-D float32 vector of length ``output_dim``.
        """
        feat_map = intermediate_maps[self.layer_index]  # (H_l, W_l, C_l)
        h_l, w_l, c_l = feat_map.shape

        # Map viewport coordinates → feature-map coordinates
        fx = mouse_x / screen_width  * w_l
        fy = mouse_y / screen_height * h_l

        attention  = self._gaussian_mask(fx, fy, h_l, w_l)            # (H, W)
        weighted   = feat_map * attention[:, :, np.newaxis]            # (H, W, C)
        local_vec  = weighted.sum(axis=(0, 1)) / (attention.sum() + 1e-8)  # (C,)

        return self._fc_forward(local_vec, c_l)

    # ------------------------------------------------------------------

    def _gaussian_mask(self, cx: float, cy: float, h: int, w: int) -> np.ndarray:
        ys = np.arange(h, dtype=np.float32)
        xs = np.arange(w, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        sigma2 = self.attention_sigma ** 2
        mask   = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2.0 * sigma2))
        return (mask / (mask.sum() + 1e-8)).astype(np.float32)

    def _fc_forward(self, vec: np.ndarray, in_dim: int) -> np.ndarray:
        if self._fc_W is None or self._fc_W.shape[0] != in_dim:
            rng          = np.random.default_rng(self._seed)
            std          = np.sqrt(2.0 / in_dim)
            self._fc_W   = rng.normal(0, std, (in_dim, self.output_dim)).astype(np.float32)
            self._fc_b   = np.zeros(self.output_dim, dtype=np.float32)
        return np.maximum(vec @ self._fc_W + self._fc_b, 0.0)

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LocalCNN("
            f"sigma={self.attention_sigma}, "
            f"out_dim={self.output_dim}, "
            f"layer_index={self.layer_index})"
        )
