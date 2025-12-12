"""Audio feature extraction utilities for UrbanSound8K."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import torch

from src import config

MIN_FRAMES = 9  # Minimum number of frames for feature stability


@dataclass
class FeatureBundle:
    """Container for both sequence and aggregated features."""

    sequence: torch.Tensor  # shape (channels, n_mfcc, time) - (1, 128, time) after update
    stats: torch.Tensor  # shape (num_features,)


class FeatureExtractor:
    """Compute MFCC-based sequence tensors and tabular statistics."""

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        n_mfcc: int = config.N_MFCC,
        n_mels: int = config.N_MELS,
        hop_length: int = config.HOP_LENGTH,
        n_fft: int = config.N_FFT,
        cache_dir: Optional[Path] = config.CACHE_DIR,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, audio_path: Path) -> Optional[Path]:
        if not self.cache_dir:
            return None
        digest = hashlib.md5(str(audio_path).encode(), usedforsecurity=False).hexdigest()
        return self.cache_dir / f"{digest}.pt"

    def _load_waveform(self, audio_path: Path) -> np.ndarray:
        """Load audio waveform matching debug.ipynb: librosa.load(file_name, sr=SAMPLE_RATE)"""
        waveform, _ = librosa.load(audio_path, sr=self.sample_rate)
        return waveform

    def _pad_feature_frames(self, feat: np.ndarray, min_frames: int = MIN_FRAMES) -> np.ndarray:
        """Pad feature frames to ensure minimum length for stability."""
        frames = feat.shape[1]
        if frames >= min_frames:
            return feat
        pad = min_frames - frames
        return np.pad(feat, ((0, 0), (0, pad)), mode="edge")

    def _sequence_features(self, waveform: np.ndarray) -> torch.Tensor:
        """Extract MFCC sequence features for CNN/LSTM models.
        Matches debug.ipynb: librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        Only passes y, sr, and n_mfcc - librosa handles the rest automatically.
        """
        # Extract MFCC exactly as in debug.ipynb - only 3 parameters
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
        )
        # Ensure minimum frames for stability
        mfcc = self._pad_feature_frames(mfcc)
        # Return as (1, n_mfcc, time_frames) - single channel MFCC sequence
        # No delta features to match debug.ipynb approach
        seq = mfcc[np.newaxis, :, :]  # Add channel dimension: (1, n_mfcc, time)
        return torch.from_numpy(seq).float()

    def _stat_features(self, waveform: np.ndarray) -> torch.Tensor:
        """Extract statistical features (MFCC mean) matching debug.ipynb approach.
        Matches debug.ipynb: librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        Then takes mean: np.mean(feature.T, axis=0)
        """
        # Extract MFCC exactly as in debug.ipynb - only 3 parameters
        feature = librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
        )
        # Take mean across time frames (exactly as in debug.ipynb: np.mean(feature.T, axis=0))
        # feature shape: (n_mfcc, time_frames)
        # Transpose to (time_frames, n_mfcc), then mean along axis=0 (time dimension)
        scaled_feature = np.mean(feature.T, axis=0)  # Result: (n_mfcc,)
        return torch.from_numpy(scaled_feature).float()

    def extract(self, audio_path: Path) -> FeatureBundle:
        cache_path = self._cache_path(audio_path)
        if cache_path and cache_path.exists():
            return FeatureBundle(**torch.load(cache_path))

        waveform = self._load_waveform(audio_path)
        sequence = self._sequence_features(waveform)
        stats = self._stat_features(waveform)

        bundle = FeatureBundle(sequence=sequence, stats=stats)
        if cache_path:
            torch.save({"sequence": sequence, "stats": stats}, cache_path)
        return bundle


