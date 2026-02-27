"""
model.py
========
CRNN-CTC architecture for monophonic Optical Music Recognition.

Architecture overview::

    Input image (B, 1, H, W)
         │
    ┌────▼────┐
    │   CNN   │   Feature extractor — reduces spatial dims, produces feature maps
    └────┬────┘
         │  (B, C', 1, W')   height collapsed to 1 via pooling
         │
    ┌────▼────┐
    │ BiLSTM  │   Sequence modelling — captures long-range dependencies
    └────┬────┘
         │  (B, W', 2·hidden)
         │
    ┌────▼────┐
    │   FC    │   Linear projection → vocab_size (incl. CTC blank)
    └────┬────┘
         │  (B, W', vocab_size)
         ▼
    log_softmax → CTCLoss

The CNN backbone must collapse the height dimension to 1 so that the feature
map at the output is a 1-D sequence along the width axis.  With an input
height of 128 px, a typical approach is to use pooling strides that divide
128 down to 1 (e.g., 5 pool layers with stride 2 → 128 / 2^5 = 4, then a
final pool of (4, 1)).

**This module provides the scaffolding and shape contract.**
The CNN/RNN details are clearly separated so you can swap architectures
(e.g., ResNet18 backbone, deeper LSTM) without touching the rest of the
pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# CNN backbone  (placeholder — replace with your own)
# ---------------------------------------------------------------------------

class CNNBackbone(nn.Module):
    """Feature extractor: (B, 1, H=128, W) → (B, cnn_out_channels, 1, W').

    The current implementation is a minimal 5-block VGG-style stack that
    collapses height to 1.  Replace with ResNet18/MobileNet once you have a
    working baseline.

    Width reduction factor:  W' ≈ W / 4   (two stride-2 pools on width,
    the remaining pools only shrink height).
    """

    def __init__(self, cnn_out_channels: int = 256, cnn_dropout: float = 0.0) -> None:
        super().__init__()
        # Each block: Conv → BN → ReLU → Dropout2d → Pool
        # Pool kernel is (h, w) — we pool height aggressively, width gently.
        drop = cnn_dropout  # shorthand
        self.features = nn.Sequential(
            # Block 1 — (1, 128, W) → (64, 64, W/2)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
            nn.MaxPool2d(kernel_size=(2, 2)),  # h/2, w/2

            # Block 2 — (64, 64, W/2) → (128, 32, W/4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
            nn.MaxPool2d(kernel_size=(2, 2)),  # h/2, w/2

            # Block 3 — (128, 32, W/4) → (256, 16, W/4)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
            nn.MaxPool2d(kernel_size=(2, 1)),  # h/2, w stays

            # Block 4 — (256, 16, W/4) → (256, 8, W/4)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
            nn.MaxPool2d(kernel_size=(2, 1)),  # h/2, w stays

            # Block 5 — (256, 8, W/4) → (cnn_out, 1, W/4)
            nn.Conv2d(256, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),  # collapse h → 1, keep w
        )

    def forward(self, x: Tensor) -> Tensor:
        """(B, 1, H, W) → (B, C, 1, W')."""
        return self.features(x)


# ---------------------------------------------------------------------------
# Full CRNN
# ---------------------------------------------------------------------------

class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network with CTC output.

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size **including** CTC blank (index 0) and pad.
    cnn_out_channels : int
        Number of feature maps at the CNN output (fed into the RNN).
    rnn_hidden : int
        Hidden size of each LSTM direction.
    rnn_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout applied between LSTM layers (ignored when ``rnn_layers == 1``).
    """

    def __init__(
        self,
        vocab_size: int,
        cnn_out_channels: int = 256,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.3,
        cnn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        # ── CNN ───────────────────────────────────────────────────────────
        self.cnn = CNNBackbone(cnn_out_channels, cnn_dropout=cnn_dropout)

        # ── RNN ───────────────────────────────────────────────────────────
        self.rnn = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # ── Classifier ────────────────────────────────────────────────────
        self.fc = nn.Linear(rnn_hidden * 2, vocab_size)  # *2 for bidirectional

    def forward(self, x: Tensor, input_widths: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            (B, 1, H, W) — batch of grayscale images.
        input_widths : Tensor | None
            (B,) — original (un-padded) widths, used to compute output
            sequence lengths after the CNN width reduction.

        Returns
        -------
        log_probs : Tensor
            (T, B, vocab_size) — log-softmax output, time-first for CTCLoss.
        output_lengths : Tensor
            (B,) — valid time-steps per sample (accounts for CNN width
            reduction and padding).
        """
        # CNN feature extraction → (B, C, 1, W')
        conv = self.cnn(x)
        b, c, h, w_prime = conv.shape
        assert h == 1, f"CNN must collapse height to 1, got {h}"

        # Squeeze height, transpose to (B, W', C)
        seq = conv.squeeze(2).permute(0, 2, 1)  # (B, W', C)

        # Pack padded sequences if widths are provided
        if input_widths is not None:
            # CNN reduces width by a factor; estimate output widths
            output_lengths = self._compute_output_lengths(input_widths, x.shape[3], w_prime)
        else:
            output_lengths = torch.full((b,), w_prime, dtype=torch.long, device=x.device)

        # Bidirectional LSTM → (B, W', 2·hidden)
        rnn_out, _ = self.rnn(seq)

        # Projection → (B, W', vocab_size)
        logits = self.fc(rnn_out)

        # CTC expects (T, B, C)
        log_probs = logits.permute(1, 0, 2).log_softmax(dim=2)

        return log_probs, output_lengths

    # -- Internal helpers ---------------------------------------------------

    @staticmethod
    def _compute_output_lengths(
        input_widths: Tensor, padded_w: int, output_w: int,
    ) -> Tensor:
        """Scale original image widths by the CNN's width reduction ratio."""
        ratio = output_w / padded_w
        return (input_widths.float() * ratio).long().clamp(min=1)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Run a forward pass with dummy data to verify shapes."""
    B, H, W = 4, 128, 800
    vocab_size = 95  # 93 tokens + blank + pad
    model = CRNN(vocab_size=vocab_size, cnn_dropout=0.2)

    x = torch.randn(B, 1, H, W)
    widths = torch.tensor([800, 700, 600, 500])

    log_probs, out_lens = model(x, widths)
    print(f"Input:        ({B}, 1, {H}, {W})")
    print(f"log_probs:    {log_probs.shape}")   # (T, B, vocab_size)
    print(f"output_lens:  {out_lens}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    _smoke_test()
