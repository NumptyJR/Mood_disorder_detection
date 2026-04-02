"""
RAVDESS Data Understanding Script
FileName: ravdess_understanding 
Authors: Joshua Schaff, Isaac Campbell, Cameron Bender 
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# CONFIG
SPEECH_DIR = "Audio_Speech_Actors_01-24"
SONG_DIR = "Audio_Song_Actors_01-24"
OUTPUT_DIR = "eda_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# RAVDESS FILENAME ENCODING
EMOTION_MAP = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"
}
INTENSITY_MAP = {1: "normal", 2: "strong"}
CHANNEL_MAP = {1: "speech", 2: "song"}
STATEMENT_MAP = {1: "Kids are talking by the door", 2: "Dogs are sitting by the door"}


def parse_ravdess_filename(filepath):
    # Parse a RAVDESS filename into its metadata components. 
    filename = os.path.basename(filepath)
    parts = filename.replace(".wav", "").split("-")

    if len(parts) != 7:
        return None  # unexpected format

    parts = [int(p) for p in parts]

    return {
        "filepath": filepath,
        "filename": filename,
        "vocal_channel": CHANNEL_MAP.get(parts[1], f"unknown({parts[1]})"),
        "emotion": EMOTION_MAP.get(parts[2], f"unknown({parts[2]})"),
        "emotion_id": parts[2],
        "intensity": INTENSITY_MAP.get(parts[3], f"unknown({parts[3]})"),
        "statement": STATEMENT_MAP.get(parts[4], f"unknown({parts[4]})"),
        "repetition": parts[5],
        "actor_id": parts[6],
        "gender": "male" if parts[6] % 2 == 1 else "female"
    }


def get_audio_info(filepath):
    # Extract audio properties from a .wav file.
    try:
        y, sr = librosa.load(filepath, sr=None)  # load at native sample rate
        duration = librosa.get_duration(y=y, sr=sr)
        return {
            "sample_rate": sr,
            "duration_sec": round(duration, 2),
            "num_samples": len(y),
            "is_valid": True
        }
    except Exception as e:
        return {
            "sample_rate": None,
            "duration_sec": None,
            "num_samples": None,
            "is_valid": False,
            "error": str(e)
        }

# DATA SOURCES

print("=" * 60)
print("DATA SOURCES")
print("=" * 60)

speech_files = sorted(glob.glob(os.path.join(SPEECH_DIR, "**", "*.wav"), recursive=True))
song_files = sorted(glob.glob(os.path.join(SONG_DIR, "**", "*.wav"), recursive=True))
all_files = speech_files + song_files

print(f"Speech folder: {len(speech_files)} .wav files")
print(f"Song folder:   {len(song_files)} .wav files")
print(f"Total:         {len(all_files)} .wav files")

# DATA DESCRIPTION — Parse all filenames into metadata
print("\n" + "=" * 60)
print("DATA DESCRIPTION")
print("=" * 60)

records = []
parse_errors = []
for f in all_files:
    parsed = parse_ravdess_filename(f)
    if parsed:
        records.append(parsed)
    else:
        parse_errors.append(f)

df = pd.DataFrame(records)
print(f"\nSuccessfully parsed: {len(df)} files")
if parse_errors:
    print(f"Parse errors: {len(parse_errors)} files")
    for e in parse_errors:
        print(f"  - {e}")

print(f"Emotions:   {sorted(df['emotion'].unique())}")
print(f"Actors:     {df['actor_id'].nunique()} (IDs {df['actor_id'].min()}–{df['actor_id'].max()})")
print(f"Genders:    {dict(df['gender'].value_counts())}")
print(f"Intensities: {dict(df['intensity'].value_counts())}")

print("\n--- Emotion counts ---")
print(df["emotion"].value_counts().sort_index().to_string())

print("\n--- Emotion × Gender ---")
print(pd.crosstab(df["emotion"], df["gender"]).to_string())

print("\n--- Emotion × Intensity ---")
print(pd.crosstab(df["emotion"], df["intensity"]).to_string())


# EXPLORATORY DATA ANALYSIS — Audio properties + visuals
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print("Extracting audio properties (give this some time, it is slow)...")

audio_info = []
for i, row in df.iterrows():
    info = get_audio_info(row["filepath"])
    audio_info.append(info)
    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{len(df)} files...")

audio_df = pd.DataFrame(audio_info)
df = pd.concat([df, audio_df], axis=1)

print(f"\n--- Audio Properties ---")
print(f"Sample rates found: {df['sample_rate'].unique()}")
print(f"Duration (seconds): min={df['duration_sec'].min()}, "
      f"max={df['duration_sec'].max()}, "
      f"mean={df['duration_sec'].mean():.2f}, "
      f"std={df['duration_sec'].std():.2f}")

# VISUALIZATIONS
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("RAVDESS Dataset — Exploratory Data Analysis", fontsize=16, fontweight="bold")

# Emotion distribution
emotion_order = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
emotion_counts = df["emotion"].value_counts().reindex(emotion_order)
colors = sns.color_palette("husl", len(emotion_order))
axes[0, 0].bar(emotion_order, emotion_counts.values, color=colors)
axes[0, 0].set_title("Emotion Distribution")
axes[0, 0].set_xlabel("Emotion")
axes[0, 0].set_ylabel("Count")
axes[0, 0].tick_params(axis="x", rotation=45)

# Emotion by gender
ct_gender = pd.crosstab(df["emotion"], df["gender"]).reindex(emotion_order)
ct_gender.plot(kind="bar", ax=axes[0, 1], color=["#55A868", "#C44E52"])
axes[0, 1].set_title("Emotion × Gender")
axes[0, 1].set_xlabel("Emotion")
axes[0, 1].set_ylabel("Count")
axes[0, 1].tick_params(axis="x", rotation=45)

# Duration distribution
axes[1, 0].hist(df["duration_sec"], bins=30, color="#4C72B0", edgecolor="white")
axes[1, 0].set_title("Audio Duration Distribution")
axes[1, 0].set_xlabel("Duration (seconds)")
axes[1, 0].set_ylabel("Count")

# Duration by emotion (box plot)
df_plot = df.copy()
df_plot["emotion"] = pd.Categorical(df_plot["emotion"], categories=emotion_order, ordered=True)
df_plot.boxplot(column="duration_sec", by="emotion", ax=axes[1, 1])
axes[1, 1].set_title("Duration by Emotion")
axes[1, 1].set_xlabel("Emotion")
axes[1, 1].set_ylabel("Duration (seconds)")
axes[1, 1].tick_params(axis="x", rotation=45)
plt.sca(axes[1, 1])
plt.title("Duration by Emotion")  # override default boxplot title

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "eda_overview.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'eda_overview.png')}")
plt.close()

# Actor-level breakdown
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
fig2.suptitle("Actor-Level Analysis", fontsize=14, fontweight="bold")

# Files per actor
actor_counts = df["actor_id"].value_counts().sort_index()
axes2[0].bar(actor_counts.index, actor_counts.values, color="#4C72B0")
axes2[0].set_title("Files per Actor")
axes2[0].set_xlabel("Actor ID")
axes2[0].set_ylabel("Count")
axes2[0].set_xticks(range(1, 25))

# Mean duration per actor
actor_dur = df.groupby("actor_id")["duration_sec"].mean().sort_index()
axes2[1].bar(actor_dur.index, actor_dur.values, color="#55A868")
axes2[1].set_title("Mean Duration per Actor")
axes2[1].set_xlabel("Actor ID")
axes2[1].set_ylabel("Duration (seconds)")
axes2[1].set_xticks(range(1, 25))

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(OUTPUT_DIR, "eda_actors.png"), dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.join(OUTPUT_DIR, 'eda_actors.png')}")
plt.close()


# DATA QUALITY ASSESSMENT
print("\n" + "=" * 60)
print("DATA QUALITY ASSESSMENT")
print("=" * 60)

invalid_files = df[df["is_valid"] == False]
print(f"\nCorrupted / unreadable files: {len(invalid_files)}")
if len(invalid_files) > 0:
    print(invalid_files[["filename", "error"]].to_string(index=False))

# Check for unexpected sample rates
sr_counts = df[df["is_valid"]]["sample_rate"].value_counts()
print(f"\nSample rate consistency:")
for sr, count in sr_counts.items():
    print(f"  {int(sr)} Hz: {count} files")
if len(sr_counts) > 1:
    print("Multiple sample rates detected — normalization needed!")
else:
    print("All files share the same sample rate.")

# Check for missing emotions
print("\nMissing emotions:")
expected_emotions_speech = set(EMOTION_MAP.values())
expected_emotions_song = set(EMOTION_MAP.values()) - {"surprised"}
actual_speech = set(df[df["vocal_channel"] == "speech"]["emotion"].unique())
actual_song = set(df[df["vocal_channel"] == "song"]["emotion"].unique())
missing_speech = expected_emotions_speech - actual_speech
missing_song = expected_emotions_song - actual_song
if missing_speech:
    print(f"  Speech missing: {missing_speech}")
else:
    print("  Speech: all expected emotions present")
if missing_song:
    print(f"  Song missing: {missing_song}")
else:
    print("  Song: all expected emotions present")

# Check for neutral + strong intensity (shouldn't exist)
neutral_strong = df[(df["emotion"] == "neutral") & (df["intensity"] == "strong")]
print(f"\nNeutral with 'strong' intensity (should be 0): {len(neutral_strong)}")

# Duration outliers
q1 = df["duration_sec"].quantile(0.25)
q3 = df["duration_sec"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = df[(df["duration_sec"] < lower) | (df["duration_sec"] > upper)]
print(f"\nDuration outliers (IQR method): {len(outliers)} files")
if len(outliers) > 0:
    print(f"  Range considered normal: {lower:.2f}–{upper:.2f} sec")
    print(outliers[["filename", "emotion", "vocal_channel", "duration_sec"]].to_string(index=False))

# Save full metadata to CSV
csv_path = os.path.join(OUTPUT_DIR, "ravdess_metadata.csv")
df.to_csv(csv_path, index=False)
print(f"\nFull metadata saved to: {csv_path}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)