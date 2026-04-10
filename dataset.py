import os
import random
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from config import *

# Helpers
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load and validate the metadata CSV. Labels depend on BINARY_MODE in config."""
    df = pd.read_csv(csv_path)
    df = df[df["is_valid"] == True].copy()

    if BINARY_MODE:
        # 0 = neutral/positive, 1 = negative affect
        df["label"] = df["emotion"].apply(
            lambda e: 1 if e in NEGATIVE_EMOTIONS else 0
        )
    else:
        # 0–7 mapped from full emotion name
        df["label"] = df["emotion"].map({e: i for i, e in enumerate(EMOTIONS)})
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

    return df


def split_by_actor(df: pd.DataFrame):
    """
    Split into train / val / test strictly by actor ID.
    This prevents speaker identity leaking into evaluation.
    """
    test_df  = df[df["actor_id"].isin(TEST_ACTORS)].copy()
    val_df   = df[df["actor_id"].isin(VAL_ACTORS)].copy()
    train_df = df[~df["actor_id"].isin(TEST_ACTORS + VAL_ACTORS)].copy()
    return train_df, val_df, test_df


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency class weights to handle any imbalance."""
    counts = train_df["label"].value_counts().sort_index().values
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)

# Spectrogram pipeline
class MelSpectrogramExtractor:
    """Converts a raw waveform tensor to a normalised log-Mel spectrogram."""

    def __init__(self):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(waveform)
        mel = self.amplitude_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        return mel


# SpecAugment
class SpecAugment:
    """Frequency and time masking — only applied during training."""

    def __init__(self):
        self.freq_masking = T.FrequencyMasking(freq_mask_param=FREQ_MASK_MAX)
        self.time_masking = T.TimeMasking(time_mask_param=TIME_MASK_MAX)

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        for _ in range(NUM_MASKS):
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)
        return spec

# Dataset
class RAVDESSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.spec_augment = SpecAugment() if augment else None
        self.target_length = int(SAMPLE_RATE * CLIP_SECONDS)

        # Use cache if CACHE_DIR is set in config and the folder exists
        self.cache_dir = getattr(__import__('config'), 'CACHE_DIR', None)
        if self.cache_dir and not os.path.isdir(self.cache_dir):
            print(f"Warning: CACHE_DIR '{self.cache_dir}' not found — falling back to live loading.")
            self.cache_dir = None

        # Only need the extractor if not using cache
        self.extractor = None if self.cache_dir else MelSpectrogramExtractor()

    def __len__(self):
        return len(self.df)

    def _load_audio(self, filepath: str) -> torch.Tensor:
        full_path = os.path.join(AUDIO_ROOT, filepath)

        data, sr = sf.read(full_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)

        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        length = waveform.shape[1]
        if length < self.target_length:
            pad = self.target_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_length]

        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        return waveform

    def _load_mel(self, filepath: str) -> torch.Tensor:
        """Load pre-computed spectrogram from cache."""
        safe_name = filepath.replace("/", "_").replace("\\", "_") + ".pt"
        return torch.load(os.path.join(self.cache_dir, safe_name), weights_only=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.cache_dir:
            # Fast path: load pre-computed spectrogram
            mel = self._load_mel(row["filepath"])
        else:
            # Slow path: compute on the fly
            waveform = self._load_audio(row["filepath"])
            if self.augment and random.random() < 0.3:
                waveform = waveform + 0.005 * torch.randn_like(waveform)
            mel = self.extractor(waveform)

        if self.augment:
            mel = self.spec_augment(mel)

        label = torch.tensor(row["label"], dtype=torch.long)
        return mel, label

# DataLoaders
def get_dataloaders(csv_path: str = METADATA_CSV):
    set_seed()
    df = load_metadata(csv_path)
    train_df, val_df, test_df = split_by_actor(df)

    print(f"Dataset split  →  train: {len(train_df)}  |  val: {len(val_df)}  |  test: {len(test_df)}")
    if BINARY_MODE:
        pos = (train_df['label'] == 0).sum()
        neg = (train_df['label'] == 1).sum()
        print(f"Train class distribution  →  neutral/positive: {pos}  |  negative: {neg}\n")
    else:
        counts = train_df['label'].value_counts().sort_index()
        print(f"Train class distribution:\n{counts.to_string()}\n")

    train_ds = RAVDESSDataset(train_df, augment=True)
    val_ds   = RAVDESSDataset(val_df,   augment=False)
    test_ds  = RAVDESSDataset(test_df,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=False)

    class_weights = compute_class_weights(train_df)
    return train_loader, val_loader, test_loader, class_weights