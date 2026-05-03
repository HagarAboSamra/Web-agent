"""
screenshot.py

Captures a webpage as a screenshot, resizes it to 210 × 160 × 3,
and returns the RGB array directly in memory as a NumPy array.

No file I/O — the array is consumed directly by image_processing.py.

Dependencies:
    pip install selenium pillow numpy chromedriver-autoinstaller
"""

import time
from io import BytesIO

import numpy as np
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_autoinstaller


# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_WIDTH  = 160
TARGET_HEIGHT = 210
TARGET_SHAPE  = (TARGET_HEIGHT, TARGET_WIDTH, 3)   # (210, 160, 3)


class WebScreenshot:
    """
    Opens a URL in a headless browser, captures a screenshot, and
    returns a uint8 RGB NumPy array of shape (210, 160, 3).

    The array lives purely in memory — use it directly or pass it to
    ``image_processing.array_to_tensor()``.

    Parameters
    ----------
    url : str
        Fully-qualified URL, e.g. ``'https://example.com'``.
    wait_seconds : float
        Seconds to wait after page load for JS to settle.  Default 2.
    viewport_width : int
        Headless browser viewport width in pixels.  Default 1280.
    viewport_height : int
        Headless browser viewport height in pixels.  Default 900.

    Usage
    -----
    >>> ws  = WebScreenshot("https://example.com")
    >>> arr = ws.capture()          # numpy uint8 (210, 160, 3)
    >>> arr = ws.rgb_array          # same array, cached after first capture
    """

    def __init__(
        self,
        url: str,
        wait_seconds: float = 2.0,
        viewport_width: int = 1280,
        viewport_height: int = 900,
    ):
        self.url             = url
        self.wait_seconds    = wait_seconds
        self.viewport_width  = viewport_width
        self.viewport_height = viewport_height
        self._rgb_array: np.ndarray | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def capture(self) -> np.ndarray:
        """
        Navigate to the URL, take a screenshot, resize to (210, 160, 3),
        cache and return the uint8 RGB array.
        """
        png_bytes       = self._take_screenshot()
        self._rgb_array = self._process_image(png_bytes)
        return self._rgb_array

    @property
    def rgb_array(self) -> np.ndarray | None:
        """Cached RGB array after ``capture()``, or None if not yet called."""
        return self._rgb_array

    # ── Private helpers ────────────────────────────────────────────────────────

    def _take_screenshot(self) -> bytes:
        chromedriver_autoinstaller.install()

        opts = Options()
        opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument(f"--window-size={self.viewport_width},{self.viewport_height}")

        driver = webdriver.Chrome(options=opts)
        try:
            driver.get(self.url)
            time.sleep(self.wait_seconds)
            png_bytes = driver.get_screenshot_as_png()
        finally:
            driver.quit()

        return png_bytes

    @staticmethod
    def _process_image(png_bytes: bytes) -> np.ndarray:
        """PNG bytes → resized uint8 RGB array of shape (210, 160, 3)."""
        image         = Image.open(BytesIO(png_bytes)).convert("RGB")
        image_resized = image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
        rgb_array     = np.array(image_resized, dtype=np.uint8)
        assert rgb_array.shape == TARGET_SHAPE, (
            f"Unexpected shape {rgb_array.shape}, expected {TARGET_SHAPE}"
        )
        return rgb_array


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    ws  = WebScreenshot(url)
    arr = ws.capture()
    print(f"Shape : {arr.shape}")
    print(f"dtype : {arr.dtype}")
    print(f"Range : [{arr.min()}, {arr.max()}]")
