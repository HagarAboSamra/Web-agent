"""
screen_processor.py
Converts a raw RGB numpy array (from Gymnasium obs) to a normalised
float32 tensor ready for GlobalCNN.

MiniWoB++ Gymnasium observations are uint8 RGB arrays of shape (H, W, 3)
at 160×210 by default.  We resize them to the requested viewport and
apply ImageNet normalisation.

Viewports:  desktop=1280x720  tablet=768x1024  mobile=375x667
"""

import numpy as np

VIEWPORTS = {
    "desktop": (1280, 720),
    "tablet":  (768, 1024),
    "mobile":  (375, 667),
    "miniwob": (160, 210),   # native MiniWoB++ pixel obs
}


class ScreenProcessor:
    """
    Resize + normalise a raw RGB observation from Gymnasium.

    Parameters
    ----------
    viewport : str | tuple[int, int]
        "desktop", "tablet", "mobile", "miniwob", or explicit (width, height).
    """

    MEANS = (0.485, 0.456, 0.406)
    STDS  = (0.229, 0.224, 0.225)

    def __init__(self, viewport="miniwob"):
        if isinstance(viewport, str):
            self.w, self.h = VIEWPORTS[viewport.lower()]
        else:
            self.w, self.h = int(viewport[0]), int(viewport[1])
        self._means = np.array(self.MEANS, dtype=np.float32)
        self._stds  = np.array(self.STDS,  dtype=np.float32)

    def process(self, rgb_array):
        """
        Convert uint8 RGB array → (W, H, 3) float32 normalised tensor.

        Parameters
        ----------
        rgb_array : np.ndarray  shape (H, W, 3) uint8  — from Gymnasium obs

        Returns
        -------
        np.ndarray  shape (W, H, 3) float32
        """
        rgb = self._resize(rgb_array["screenshot"].astype(np.uint8))  # (H, W, 3)
        rgb = rgb.transpose(1, 0, 2)                      # → (W, H, 3)
        return (rgb.astype(np.float32) / 255.0 - self._means) / self._stds

    def _resize(self, pixels):
        from PIL import Image
        h, w = pixels.shape[:2]
        if (w, h) == (self.w, self.h):
            return pixels
        img = Image.fromarray(pixels).resize((self.w, self.h), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)
