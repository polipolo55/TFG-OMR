"""
CRNN-CTC package for Optical Music Recognition.

This package implements the full training and inference pipeline for a
Convolutional Recurrent Neural Network trained with CTC loss on monophonic
staff-line images, using LMX (Linearized MusicXML) as the target encoding.

Modules
-------
config      Centralised hyperparameters and paths.
vocab       LMX token ↔ integer mapping (CTC blank at index 0).
dataset     PyTorch Dataset / DataLoader / collation for PNG + LMX pairs.
model       CRNN architecture: CNN backbone + bidirectional LSTM + FC head.
train       Training loop with AMP, OneCycleLR, CTC loss, checkpointing.
evaluate    Greedy CTC decoding, SER metric, full evaluation loop.
"""

from .config import Config
from .vocab import Vocabulary

__all__ = ["Config", "Vocabulary"]
