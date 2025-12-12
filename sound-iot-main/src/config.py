"""Central configuration for UrbanSound8K experiments."""
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent.parent / "archive"
METADATA_CSV = DATASET_ROOT / "UrbanSound8K.csv"
SAMPLE_RATE = 22050
N_MFCC = 128  # Updated to match debug.ipynb
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 30
LEARNING_RATE = 1e-3
CACHE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "cache"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "checkpoints"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "results"
SEED = 1337

CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]
