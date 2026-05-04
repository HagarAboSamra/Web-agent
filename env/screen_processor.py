"""
screen_processor.py

Handles raw visual input captured by Playwright from a live browser session.

Instead of a fixed-resolution MiniWoB array, this module accepts PNG screenshot
bytes produced by ``playwright.Page.screenshot()``, decodes them, resizes to the
configured viewport, and normalises for the CNN backbone.

Supported viewport presets (width × height)
--------------------------------------------
"desktop"  : 1280 × 720
"tablet"   :  768 × 1024
"mobile"   :  375 ×  667

A custom (width, height) tuple may also be passed directly.
"""

import io
import numpy as np
from typing import Tuple


VIEWPORT_PRESETS: dict[str, Tuple[int, int]] = {
    "desktop": (1280, 720),
    "tablet":  (768,  1024),
    "mobile":  (375,  667),
}


class ScreenProcessor:
    """
    Decodes a Playwright PNG screenshot and produces a normalised float32
    tensor ready for the CNN.

    Parameters
    ----------
    viewport : str | tuple[int, int]
        Named preset (``"desktop"``, ``"tablet"``, ``"mobile"``) or an
        explicit ``(width, height)`` tuple.  Screenshots are resized to
        this resolution before normalisation.
    channel_means : tuple[float, float, float] | None
        Per-channel (R, G, B) means.  Defaults to ImageNet means.
    channel_stds : tuple[float, float, float] | None
        Per-channel (R, G, B) standard deviations.  Defaults to ImageNet.

    Example
    -------
    >>> proc = ScreenProcessor(viewport="desktop")
    >>> png_bytes = page.screenshot()          # from Playwright Page
    >>> tensor = proc.process(png_bytes)       # (1280, 720, 3) float32
    """

    _DEFAULT_MEANS: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    _DEFAULT_STDS:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        viewport:      str | Tuple[int, int] = "desktop",
        channel_means: Tuple[float, float, float] | None = None,
        channel_stds:  Tuple[float, float, float] | None = None,
    ) -> None:
        if isinstance(viewport, str):
            key = viewport.lower()
            if key not in VIEWPORT_PRESETS:
                raise ValueError(
                    f"Unknown viewport preset '{viewport}'. "
                    f"Choose from {list(VIEWPORT_PRESETS)} or pass (W, H)."
                )
            self.expected_width, self.expected_height = VIEWPORT_PRESETS[key]
            self.viewport_name = key
        else:
            self.expected_width  = int(viewport[0])
            self.expected_height = int(viewport[1])
            self.viewport_name   = f"custom({self.expected_width}x{self.expected_height})"

        self.means = np.array(channel_means or self._DEFAULT_MEANS, dtype=np.float32)
        self.stds  = np.array(channel_stds  or self._DEFAULT_STDS,  dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, png_bytes: bytes) -> np.ndarray:
        """
        Decode, resize, and normalise a Playwright PNG screenshot.

        Parameters
        ----------
        png_bytes : bytes
            Raw PNG data from ``playwright.Page.screenshot()``.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(W, H, 3)`` normalised per-channel.
        """
        pixels = self._decode_png(png_bytes)    # (H_raw, W_raw, 3) uint8
        pixels = self._resize(pixels)           # (H, W, 3)         uint8
        pixels = pixels.transpose(1, 0, 2)      # → (W, H, 3)
        return self._normalize(pixels)

    def playwright_viewport(self) -> dict:
        """
        Returns a Playwright-compatible viewport dict.

        Example
        -------
        >>> page = browser.new_page(viewport=proc.playwright_viewport())
        """
        return {"width": self.expected_width, "height": self.expected_height}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_png(png_bytes: bytes) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("pip install pillow") from exc
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def _resize(self, pixels: np.ndarray) -> np.ndarray:
        from PIL import Image
        h, w = pixels.shape[:2]
        tw, th = self.expected_width, self.expected_height
        if (w, h) == (tw, th):
            return pixels
        return np.array(
            Image.fromarray(pixels).resize((tw, th), Image.LANCZOS),
            dtype=np.uint8,
        )

    def _normalize(self, pixels: np.ndarray) -> np.ndarray:
        scaled = pixels.astype(np.float32) / 255.0
        return (scaled - self.means) / self.stds

    # ------------------------------------------------------------------

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        return (self.expected_width, self.expected_height, 3)

    def __repr__(self) -> str:
        return (
            f"ScreenProcessor("
            f"viewport={self.viewport_name!r}, "
            f"resolution={self.expected_width}×{self.expected_height})"
        )
