"""
local_cnn.py

Local_CNN: uses the spatial feature map from Global_CNN together with the
extracted DOM elements to predict the precise pixel location of each element
and decide whether interaction requires a cursor click or keyboard input.

Design rationale
----------------
Local_CNN does NOT inherit from Global_CNN because they have different roles:
  - Global_CNN   : image → compact global representation (screenshot-level)
  - Local_CNN    : (global feature map, DOM node) → element-level localisation

Instead, Local_CNN *composes* a Global_CNN instance and wraps it.  This keeps
responsibilities clean and allows Global_CNN weights to be frozen or fine-tuned
independently.

For each DOM node the model produces:
  predicted_cx   – horizontal centre in screenshot pixels  (0 … 160)
  predicted_cy   – vertical centre in screenshot pixels    (0 … 210)
  interaction    – 0=none, 1=click (cursor), 2=type (keyboard)
  confidence     – scalar in [0, 1]

Dependencies:
    pip install torch numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from image_processing import Global_CNN, array_to_tensor
from DOM_elements import DOMNode

# Interaction type constants
INTERACTION_NONE     = 0
INTERACTION_CLICK    = 1   # cursor
INTERACTION_KEYBOARD = 2   # keyboard / type


# ── Prediction result container ───────────────────────────────────────────────

@dataclass
class ElementPrediction:
    """
    Localisation + interaction prediction for a single DOM node.

    Attributes
    ----------
    node          : DOMNode        – the source DOM element
    predicted_cx  : float          – predicted centre-x in screenshot px
    predicted_cy  : float          – predicted centre-y in screenshot px
    interaction   : int            – INTERACTION_NONE / CLICK / KEYBOARD
    confidence    : float          – model confidence in [0, 1]
    feature_vec   : np.ndarray     – the raw 395-dim feature fed to the head
    """
    node:         DOMNode
    predicted_cx: float
    predicted_cy: float
    interaction:  int
    confidence:   float
    feature_vec:  Optional[np.ndarray] = None

    @property
    def interaction_label(self) -> str:
        return {0: "none", 1: "click", 2: "keyboard"}.get(self.interaction, "?")

    def __repr__(self) -> str:
        return (
            f"ElementPrediction("
            f"tag={self.node.tag!r}, "
            f"cx={self.predicted_cx:.1f}, cy={self.predicted_cy:.1f}, "
            f"action={self.interaction_label}, conf={self.confidence:.3f})"
        )


# ── DOM node encoder (pure torch, no CNN) ─────────────────────────────────────

class _DOMEncoder(nn.Module):
    """
    Encodes a DOM node's scalar/binary properties into an 11-dim vector that
    can be concatenated with the global CNN features.

    Feature layout (mirrors feature_map.py conventions):
      0  is_visible
      1  is_clickable
      2  is_typeable
      3  normalised_cx      (bbox centre-x / 160)
      4  normalised_cy      (bbox centre-y / 210)
      5  normalised_w       (bbox width    / 160)
      6  normalised_h       (bbox height   / 210)
      7  tag_click_hint     (1 if tag in clickable set)
      8  tag_type_hint      (1 if tag in typeable set)
      9  has_text           (1 if node.text is non-empty)
     10  attr_count_norm    (min(num_attrs, 20) / 20)
    """

    DIM = 11
    _CLICK_TAGS = {"a", "button", "select", "option", "label", "summary"}
    _TYPE_TAGS  = {"input", "textarea"}

    @staticmethod
    def encode(node: DOMNode, img_w: int = 160, img_h: int = 210) -> torch.Tensor:
        """Return a float32 tensor of shape (DOMEncoder.DIM,)."""
        bbox = node.bbox
        vec = torch.tensor(
            [
                float(node.is_visible),
                float(node.is_clickable),
                float(node.is_typeable),
                bbox.get("x", 0) / img_w,
                bbox.get("y", 0) / img_h,
                bbox.get("width",  0) / img_w,
                bbox.get("height", 0) / img_h,
                float(node.tag in _DOMEncoder._CLICK_TAGS),
                float(node.tag in _DOMEncoder._TYPE_TAGS),
                float(bool(node.text)),
                min(len(node.attributes), 20) / 20.0,
            ],
            dtype=torch.float32,
        )
        return vec


# ── Local CNN head ────────────────────────────────────────────────────────────

class _LocalHead(nn.Module):
    """
    Lightweight prediction head.

    Input : (B, 384 + 11) = (B, 395)
    Output: (B, 4)  →  [cx_norm, cy_norm, interaction_logits×2, confidence_logit]

    We predict cx / cy as values in [0, 1] (normalised to screenshot dims).
    """

    def __init__(self, in_features: int = 395, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 5),   # cx, cy, int_logit_click, int_logit_key, conf
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# ── Local CNN (main public class) ─────────────────────────────────────────────

class Local_CNN(nn.Module):
    """
    Combines Global_CNN's spatial representation with per-element DOM features
    to predict the exact on-screen location and required interaction type for
    each DOM node.

    Parameters
    ----------
    global_cnn : Global_CNN
        Pre-built (optionally pre-trained) Global_CNN instance.
        Its weights can be frozen via ``freeze_backbone=True``.
    freeze_backbone : bool
        If True, Global_CNN weights are not updated during Local_CNN training.
    img_width : int
        Screenshot width in pixels (default 160, matching Global_CNN input).
    img_height : int
        Screenshot height in pixels (default 210).
    """

    # Screenshot dimensions must match Global_CNN expectations
    _IMG_W = 160
    _IMG_H = 210

    def __init__(
        self,
        global_cnn: Global_CNN,
        freeze_backbone: bool = False,
        img_width:  int = 160,
        img_height: int = 210,
    ):
        super().__init__()
        self.global_cnn     = global_cnn
        self.img_width      = img_width
        self.img_height     = img_height

        if freeze_backbone:
            for param in self.global_cnn.parameters():
                param.requires_grad = False

        # Feature dim: 384 (Global_CNN FC1) + 11 (DOM encoder)
        self.head = _LocalHead(in_features=384 + _DOMEncoder.DIM)

    # ── Forward (batch mode, for training) ────────────────────────────────────

    def forward(
        self,
        screenshot_tensor: torch.Tensor,      # (B, 3, 210, 160)
        dom_feat_tensor:   torch.Tensor,       # (B, 11)
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        screenshot_tensor : (B, 3, H, W)
        dom_feat_tensor   : (B, DOMEncoder.DIM)

        Returns
        -------
        dict with keys:
            'cx'            : (B,)   normalised [0,1]
            'cy'            : (B,)   normalised [0,1]
            'interaction'   : (B, 2) logits for [click, keyboard]
            'confidence'    : (B,)   raw logit (apply sigmoid for probability)
        """
        global_feat = self.global_cnn.extract_features(screenshot_tensor)  # (B, 384)
        combined    = torch.cat([global_feat, dom_feat_tensor], dim=1)      # (B, 395)
        raw         = self.head(combined)                                   # (B, 5)

        return {
            "cx":          torch.sigmoid(raw[:, 0]),          # normalised cx
            "cy":          torch.sigmoid(raw[:, 1]),          # normalised cy
            "interaction": raw[:, 2:4],                       # logits (B, 2)
            "confidence":  raw[:, 4],                         # raw logit
        }

    # ── Inference (predict for a list of DOM nodes) ───────────────────────────

    @torch.no_grad()
    def predict(
        self,
        screenshot_tensor: torch.Tensor,      # (1, 3, 210, 160)
        dom_nodes: list[DOMNode],
    ) -> list[ElementPrediction]:
        """
        Predict location and interaction type for every DOM node.

        Parameters
        ----------
        screenshot_tensor : torch.Tensor
            Preprocessed screenshot.  Use ``array_to_tensor()`` from
            image_processing.py to build this from a (210,160,3) uint8 array.
        dom_nodes : list[DOMNode]

        Returns
        -------
        list[ElementPrediction]
            One prediction per node, in the same order as ``dom_nodes``.
        """
        self.eval()
        predictions: list[ElementPrediction] = []

        # Repeat screenshot once per node
        B = len(dom_nodes)
        if B == 0:
            return []

        screen_batch = screenshot_tensor.expand(B, -1, -1, -1)   # (B, 3, H, W)

        # Build DOM feature batch
        dom_vecs = torch.stack(
            [_DOMEncoder.encode(n, self.img_width, self.img_height) for n in dom_nodes]
        )                                                         # (B, 11)

        out = self.forward(screen_batch, dom_vecs)

        cx_arr   = out["cx"].cpu().numpy()            # (B,)
        cy_arr   = out["cy"].cpu().numpy()            # (B,)
        int_prob = F.softmax(out["interaction"], dim=1).cpu().numpy()  # (B, 2)
        conf_arr = torch.sigmoid(out["confidence"]).cpu().numpy()      # (B,)

        for i, node in enumerate(dom_nodes):
            # Interaction: compare click vs keyboard probabilities
            click_prob    = float(int_prob[i, 0])
            keyboard_prob = float(int_prob[i, 1])

            if not node.is_visible:
                interaction = INTERACTION_NONE
            elif node.is_typeable and keyboard_prob > click_prob:
                interaction = INTERACTION_KEYBOARD
            else:
                interaction = INTERACTION_CLICK if node.is_clickable else INTERACTION_NONE

            feat_vec = np.concatenate([
                out["cx"][i:i+1].cpu().numpy(),
                out["cy"][i:i+1].cpu().numpy(),
            ])

            predictions.append(ElementPrediction(
                node         = node,
                predicted_cx = float(cx_arr[i]) * self.img_width,
                predicted_cy = float(cy_arr[i]) * self.img_height,
                interaction  = interaction,
                confidence   = float(conf_arr[i]),
                feature_vec  = feat_vec,
            ))

        return predictions

    # ── Convenience factory ────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        num_logits: int = 128,
        freeze_backbone: bool = False,
    ) -> "Local_CNN":
        """Create a Local_CNN with a fresh (untrained) Global_CNN backbone."""
        backbone = Global_CNN(num_logits=num_logits)
        return cls(backbone, freeze_backbone=freeze_backbone)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"[Local_CNN] Saved → {path}")

    def load(self, path: str, map_location: str = "cpu"):
        self.load_state_dict(torch.load(path, map_location=map_location))
        print(f"[Local_CNN] Loaded ← {path}")


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from DOM_elements import dom_elements as DOMExtractor
    from image_processing import screenshot_to_tensor

    url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"

    # ── Capture screenshot directly into tensor ───────────────────────────────
    print(f"[local_cnn] Capturing screenshot from {url} …")
    _, tensor = screenshot_to_tensor(url)

    # ── Extract DOM nodes ─────────────────────────────────────────────────────
    print(f"[local_cnn] Extracting DOM from {url} …")
    extractor = DOMExtractor(url)
    nodes     = extractor.extract()
    print(f"[local_cnn] {len(nodes)} nodes extracted.")

    # ── Build Local_CNN and predict ───────────────────────────────────────────
    model       = Local_CNN.build(num_logits=128, freeze_backbone=False)
    predictions = model.predict(tensor, nodes)

    print(f"\n=== Top 15 predictions ===")
    sorted_preds = sorted(predictions, key=lambda p: p.confidence, reverse=True)
    for pred in sorted_preds[:15]:
        print(
            f"  [{pred.interaction_label:8s}] "
            f"cx={pred.predicted_cx:6.1f}  cy={pred.predicted_cy:6.1f}  "
            f"conf={pred.confidence:.3f}  {pred.node.tag:<10}  "
            f"{pred.node.text[:40]!r}"
        )

    # ── Cursor vs keyboard summary ────────────────────────────────────────────
    n_click    = sum(1 for p in predictions if p.interaction == INTERACTION_CLICK)
    n_keyboard = sum(1 for p in predictions if p.interaction == INTERACTION_KEYBOARD)
    n_none     = sum(1 for p in predictions if p.interaction == INTERACTION_NONE)
    print(f"\nInteraction summary: click={n_click}  keyboard={n_keyboard}  none={n_none}")
