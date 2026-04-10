import os
import hashlib
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import torchaudio.transforms as T
from tqdm import tqdm

from config import (
    METADATA_CSV, AUDIO_ROOT, SAMPLE_RATE, CLIP_SECONDS,
    N_MELS, N_FFT, HOP_LENGTH, F_MIN, F_MAX
)


def cache_dir_name() -> str:
    key = f"{SAMPLE_RATE}_{CLIP_SECONDS}_{N_MELS}_{N_FFT}_{HOP_LENGTH}_{F_MIN}_{F_MAX}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return os.path.join("cache", f"mel_{h}")


def load_audio(filepath: str, target_length: int) -> torch.Tensor:
    full_path = os.path.join(AUDIO_ROOT, filepath)
    data, sr = sf.read(full_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)

    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    length = waveform.shape[1]
    if length < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - length))
    else:
        waveform = waveform[:, :target_length]

    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val

    return waveform


def build_cache():
    cache_dir = cache_dir_name()
    os.makedirs(cache_dir, exist_ok=True)

    # Write a human-readable config summary next to the cache
    with open(os.path.join(cache_dir, "config.txt"), "w") as f:
        f.write(f"SAMPLE_RATE={SAMPLE_RATE}\nCLIP_SECONDS={CLIP_SECONDS}\n"
                f"N_MELS={N_MELS}\nN_FFT={N_FFT}\nHOP_LENGTH={HOP_LENGTH}\n"
                f"F_MIN={F_MIN}\nF_MAX={F_MAX}\n")

    df = pd.read_csv(METADATA_CSV)
    df = df[df["is_valid"] == True].reset_index(drop=True)

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=2.0,
    )
    amplitude_to_db = T.AmplitudeToDB(top_db=80)
    target_length = SAMPLE_RATE * CLIP_SECONDS

    skipped, processed = 0, 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building cache"):
        # Use a safe filename derived from the original filepath
        safe_name = row["filepath"].replace("/", "_").replace("\\", "_") + ".pt"
        out_path = os.path.join(cache_dir, safe_name)

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            waveform = load_audio(row["filepath"], target_length)
            mel = mel_transform(waveform)
            mel = amplitude_to_db(mel)
            mel = (mel - mel.mean()) / (mel.std() + 1e-9)
            torch.save(mel, out_path)
            processed += 1
        except Exception as e:
            print(f"\nSkipping {row['filepath']}: {e}")

    print(f"\nDone. Processed: {processed}  |  Already cached: {skipped}")
    print(f"Cache location: {cache_dir}")
    print(f"\nAdd this to config.py:\n  CACHE_DIR = \"{cache_dir}\"")


if __name__ == "__main__":
    build_cache()