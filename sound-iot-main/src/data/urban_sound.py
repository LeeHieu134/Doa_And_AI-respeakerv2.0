"""PyTorch dataset and data-loading helpers for UrbanSound8K."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src import config
from src.features.extractor import FeatureBundle, FeatureExtractor


@dataclass
class UrbanSoundSample:
    path: Path
    label: int


class UrbanSoundDataset(Dataset):
    """Returns both sequence and aggregated features for a slice."""

    def __init__(self, samples: Sequence[UrbanSoundSample], extractor: FeatureExtractor) -> None:
        self.samples = list(samples)
        self.extractor = extractor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[FeatureBundle, int]:
        sample = self.samples[idx]
        features = self.extractor.extract(sample.path)
        return features, sample.label


def load_metadata(meta_csv: Path = config.METADATA_CSV) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    df["slice_file_name"] = df["slice_file_name"].astype(str)
    return df


def build_samples_for_folds(folds: Sequence[int]) -> List[UrbanSoundSample]:
    df = load_metadata()
    subset = df[df["fold"].isin(folds)]
    samples: List[UrbanSoundSample] = []
    for _, row in subset.iterrows():
        audio_path = config.DATASET_ROOT / f"fold{row['fold']}" / row["slice_file_name"]
        samples.append(UrbanSoundSample(path=audio_path, label=int(row["classID"])))
    return samples


def collate_batch(batch):
    sequences = [item[0].sequence for item in batch]
    stats = [item[0].stats for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    seq_lengths = [seq.shape[-1] for seq in sequences]
    max_len = max(seq_lengths)
    padded_sequences = []
    for seq in sequences:
        pad_width = max_len - seq.shape[-1]
        if pad_width > 0:
            # pad_tensor must have same number of dimensions as seq
            # seq shape: (channels, n_mfcc, time_frames) = (1, 128, time) after extractor update
            # pad_tensor needs shape: (1, 128, pad_width) to match seq dimensions
            pad_shape = list(seq.shape)
            pad_shape[-1] = pad_width  # Replace last dimension with pad_width
            pad_tensor = torch.zeros(pad_shape, dtype=seq.dtype, device=seq.device)
            seq = torch.cat([seq, pad_tensor], dim=-1)
        padded_sequences.append(seq)
    sequence_batch = torch.stack(padded_sequences, dim=0)
    stats_batch = torch.stack(stats, dim=0)
    return {"sequence": sequence_batch, "stats": stats_batch}, labels


def create_dataloaders(
    extractor: FeatureExtractor,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Return train/test loaders using train_test_split with stratification.
    
    Args:
        extractor: Feature extractor instance
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        test_size: Test/validation split ratio (default 0.2)
        random_state: Random seed for train_test_split (default 42)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load all samples from all folds
    all_samples = build_samples_for_folds(range(1, 11))
    # Split using train_test_split with stratification
    labels = [s.label for s in all_samples]
    train_samples, val_samples = train_test_split(
        all_samples,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    
    train_ds = UrbanSoundDataset(train_samples, extractor)
    val_ds = UrbanSoundDataset(val_samples, extractor)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    return train_loader, val_loader


