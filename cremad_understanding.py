"""
CREMA-D + Combined Data Understanding Script
FileName: cremad_understanding
Authors: Joshua Schaff, Isaac Campbell, Cameron Bender
"""

import os
import glob
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

AUDIO_DIR = "AudioWAV"
OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CREMA-D FILENAME ENCODING
EMOTION_MAP = {
    "ANG": ("angry",   5),
    "DIS": ("disgust", 7),
    "FEA": ("fearful", 6),
    "HAP": ("happy",   3),
    "NEU": ("neutral", 1),
    "SAD": ("sad",     4),
}
INTENSITY_MAP = {
    "XX": "normal",
    "LO": "normal",
    "MD": "normal",
    "HI": "strong",
    "X":  "normal",
}
STATEMENT_MAP = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be fixed",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I'm not sure I can do it",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

MALE_ACTORS = {
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
    1012, 1013, 1014, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
    1023, 1024, 1025, 1026, 1027, 1028, 1030, 1032, 1033, 1034,
    1035, 1036, 1037, 1039, 1040, 1041, 1042, 1043, 1044, 1046,
    1047, 1048, 1049, 1050, 1051, 1052, 1054, 1055,
}


def parse_cremad_filename(filepath):
    filename = os.path.basename(filepath)
    parts = filename.replace(".wav", "").split("_")

    if len(parts) != 4:
        return None

    actor_raw, sentence_code, emotion_code, intensity_code = parts

    try:
        actor_raw = int(actor_raw)
    except ValueError:
        return None

    if emotion_code not in EMOTION_MAP:
        return None

    emotion_label, emotion_id = EMOTION_MAP[emotion_code]

    return {
        "filepath": os.path.join(AUDIO_DIR, filename),
        "filename": filename,
        "vocal_channel": "speech",
        "emotion": emotion_label,
        "emotion_id": emotion_id,
        "intensity": INTENSITY_MAP.get(intensity_code, "normal"),
        "statement": STATEMENT_MAP.get(sentence_code, sentence_code),
        "repetition": 1,
        "actor_id": actor_raw - 976,   # remap 1001–1091 → 25–115
        "gender": "male" if actor_raw in MALE_ACTORS else "female",
    }


def get_audio_info(filepath):
    try:
        y, sr = librosa.load(filepath, sr=None)
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

all_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
print(f"AudioWAV folder: {len(all_files)} .wav files")

# DATA DESCRIPTION
print("\n" + "=" * 60)
print("DATA DESCRIPTION")
print("=" * 60)

records = []
parse_errors = []
for f in all_files:
    parsed = parse_cremad_filename(f)
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

print(f"Emotions:    {sorted(df['emotion'].unique())}")
print(f"Actors:      {df['actor_id'].nunique()} (IDs {df['actor_id'].min()}–{df['actor_id'].max()})")
print(f"Genders:     {dict(df['gender'].value_counts())}")
print(f"Intensities: {dict(df['intensity'].value_counts())}")

print("\n--- Emotion counts ---")
print(df["emotion"].value_counts().sort_index().to_string())

print("\n--- Emotion × Gender ---")
print(pd.crosstab(df["emotion"], df["gender"]).to_string())

print("\n--- Emotion × Intensity ---")
print(pd.crosstab(df["emotion"], df["intensity"]).to_string())

# EXPLORATORY DATA ANALYSIS
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print("Extracting audio properties (give this some time, it is slow)...")

audio_info = []
for i, row in df.iterrows():
    info = get_audio_info(row["filepath"])
    audio_info.append(info)
    if (i + 1) % 500 == 0:
        print(f"  Processed {i + 1}/{len(df)} files...")

audio_df = pd.DataFrame(audio_info)
df = pd.concat([df, audio_df], axis=1)

print(f"\n--- Audio Properties ---")
valid = df[df["is_valid"] == True]
print(f"Sample rates found: {valid['sample_rate'].unique()}")
print(f"Duration (seconds): min={valid['duration_sec'].min()}, "
      f"max={valid['duration_sec'].max()}, "
      f"mean={valid['duration_sec'].mean():.2f}, "
      f"std={valid['duration_sec'].std():.2f}")

# VISUALIZATIONS
CREMAD_EMOTION_ORDER = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("CREMA-D Dataset — Exploratory Data Analysis", fontsize=16, fontweight="bold")

emotion_counts = valid["emotion"].value_counts().reindex(CREMAD_EMOTION_ORDER)
colors = sns.color_palette("husl", len(CREMAD_EMOTION_ORDER))
axes[0, 0].bar(CREMAD_EMOTION_ORDER, emotion_counts.values, color=colors)
axes[0, 0].set_title("Emotion Distribution")
axes[0, 0].set_xlabel("Emotion")
axes[0, 0].set_ylabel("Count")
axes[0, 0].tick_params(axis="x", rotation=45)

ct_gender = pd.crosstab(valid["emotion"], valid["gender"]).reindex(CREMAD_EMOTION_ORDER)
ct_gender.plot(kind="bar", ax=axes[0, 1], color=["#55A868", "#C44E52"])
axes[0, 1].set_title("Emotion × Gender")
axes[0, 1].set_xlabel("Emotion")
axes[0, 1].set_ylabel("Count")
axes[0, 1].tick_params(axis="x", rotation=45)

axes[1, 0].hist(valid["duration_sec"], bins=30, color="#4C72B0", edgecolor="white")
axes[1, 0].set_title("Audio Duration Distribution")
axes[1, 0].set_xlabel("Duration (seconds)")
axes[1, 0].set_ylabel("Count")

df_plot = valid.copy()
df_plot["emotion"] = pd.Categorical(df_plot["emotion"], categories=CREMAD_EMOTION_ORDER, ordered=True)
df_plot.boxplot(column="duration_sec", by="emotion", ax=axes[1, 1])
axes[1, 1].set_title("Duration by Emotion")
axes[1, 1].set_xlabel("Emotion")
axes[1, 1].set_ylabel("Duration (seconds)")
axes[1, 1].tick_params(axis="x", rotation=45)
plt.sca(axes[1, 1])
plt.title("Duration by Emotion")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "cremad_eda_overview.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'cremad_eda_overview.png')}")
plt.close()

# Actor breakdown
fig2, axes2 = plt.subplots(1, 2, figsize=(20, 5))
fig2.suptitle("CREMA-D Actor-Level Analysis", fontsize=14, fontweight="bold")

actor_counts = valid["actor_id"].value_counts().sort_index()
axes2[0].bar(actor_counts.index, actor_counts.values, color="#4C72B0")
axes2[0].set_title("Files per Actor")
axes2[0].set_xlabel("Actor ID")
axes2[0].set_ylabel("Count")
cremad_ids = sorted(valid["actor_id"].unique())
axes2[0].set_xticks([x for x in cremad_ids if x % 5 == 0 or x == cremad_ids[0]])
axes2[0].tick_params(axis="x", rotation=45, labelsize=8)

actor_dur = valid.groupby("actor_id")["duration_sec"].mean().sort_index()
axes2[1].bar(actor_dur.index, actor_dur.values, color="#55A868")
axes2[1].set_title("Mean Duration per Actor")
axes2[1].set_xlabel("Actor ID")
axes2[1].set_ylabel("Duration (seconds)")
axes2[1].set_xticks([x for x in cremad_ids if x % 5 == 0 or x == cremad_ids[0]])
axes2[1].tick_params(axis="x", rotation=45, labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(OUTPUT_DIR, "cremad_eda_actors.png"), dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.join(OUTPUT_DIR, 'cremad_eda_actors.png')}")
plt.close()

# Intensity breakdown
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle("CREMA-D Intensity Analysis", fontsize=14, fontweight="bold")

intensity_counts = valid["intensity"].value_counts()
axes3[0].bar(intensity_counts.index, intensity_counts.values, color=["#4C72B0", "#DD8452"])
axes3[0].set_title("Overall Intensity Distribution")
axes3[0].set_xlabel("Intensity")
axes3[0].set_ylabel("Count")

ct_intensity = pd.crosstab(valid["emotion"], valid["intensity"]).reindex(CREMAD_EMOTION_ORDER)
ct_intensity.plot(kind="bar", ax=axes3[1], color=["#4C72B0", "#DD8452"])
axes3[1].set_title("Emotion × Intensity")
axes3[1].set_xlabel("Emotion")
axes3[1].set_ylabel("Count")
axes3[1].tick_params(axis="x", rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(OUTPUT_DIR, "cremad_eda_intensity.png"), dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.join(OUTPUT_DIR, 'cremad_eda_intensity.png')}")
plt.close()

# DATA QUALITY ASSESSMENT
print("\n" + "=" * 60)
print("DATA QUALITY ASSESSMENT")
print("=" * 60)

invalid_files = df[df["is_valid"] == False]
print(f"\nCorrupted / unreadable files: {len(invalid_files)}")
if len(invalid_files) > 0:
    print(invalid_files[["filename", "error"]].to_string(index=False))

sr_counts = valid["sample_rate"].value_counts()
print(f"\nSample rate consistency:")
for sr, count in sr_counts.items():
    print(f"  {int(sr)} Hz: {count} files")
if len(sr_counts) > 1:
    print("Multiple sample rates detected — normalization needed!")
else:
    print("All files share the same sample rate.")

print("\nMissing emotions:")
expected = set(EMOTION_MAP.keys())
actual = set(df["emotion"].unique())
missing = {EMOTION_MAP[k][0] for k in expected} - actual
print("  None" if not missing else f"  Missing: {missing}")

neutral_strong = valid[(valid["emotion"] == "neutral") & (valid["intensity"] == "strong")]
print(f"\nNeutral with 'strong' intensity (should be 0): {len(neutral_strong)}")

q1 = valid["duration_sec"].quantile(0.25)
q3 = valid["duration_sec"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers = valid[(valid["duration_sec"] < lower) | (valid["duration_sec"] > upper)]
print(f"\nDuration outliers (IQR method): {len(outliers)} files")
if len(outliers) > 0:
    print(f"  Range considered normal: {lower:.2f}–{upper:.2f} sec")

# SAVE CREMA-D CSV
csv_path = os.path.join(OUTPUT_DIR, "cremad_metadata.csv")
df.to_csv(csv_path, index=False)
print(f"\nFull metadata saved to: {csv_path}")

# COMBINED CSV
print("\n" + "=" * 60)
print("COMBINED CSV")
print("=" * 60)

ravdess = pd.read_csv(os.path.join(OUTPUT_DIR, "ravdess_metadata.csv"))
ravdess_valid = ravdess[ravdess["is_valid"] == True]
cremad_valid = df[df["is_valid"] == True]

combined = pd.concat([ravdess_valid, cremad_valid], ignore_index=True)
combined_path = os.path.join(OUTPUT_DIR, "combined_metadata.csv")
combined.to_csv(combined_path, index=False)
print(f"RAVDESS valid:  {len(ravdess_valid)}")
print(f"CREMA-D valid:  {len(cremad_valid)}")
print(f"Combined total: {len(combined)}")
print(f"Saved to: {combined_path}")

# COMBINED VISUALIZATIONS
print("\n" + "=" * 60)
print("COMBINED DATA DESCRIPTION")
print("=" * 60)
print(f"Total files:  {len(combined)}")
print(f"Emotions:     {sorted(combined['emotion'].unique())}")
print(f"Actors:       {combined['actor_id'].nunique()} (IDs {combined['actor_id'].min()}–{combined['actor_id'].max()})")
print(f"Genders:      {dict(combined['gender'].value_counts())}")
print(f"Intensities:  {dict(combined['intensity'].value_counts())}")

print("\n--- Emotion counts ---")
print(combined["emotion"].value_counts().sort_index().to_string())

print("\n--- Emotion × Gender ---")
print(pd.crosstab(combined["emotion"], combined["gender"]).to_string())

print("\n--- Emotion × Intensity ---")
print(pd.crosstab(combined["emotion"], combined["intensity"]).to_string())

COMBINED_EMOTION_ORDER = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle("RAVDESS + CREMA-D Combined — Exploratory Data Analysis", fontsize=16, fontweight="bold")

emotion_counts_comb = combined["emotion"].value_counts().reindex(COMBINED_EMOTION_ORDER)
colors_comb = sns.color_palette("husl", len(COMBINED_EMOTION_ORDER))
axes4[0, 0].bar(COMBINED_EMOTION_ORDER, emotion_counts_comb.values, color=colors_comb)
axes4[0, 0].set_title("Emotion Distribution")
axes4[0, 0].set_xlabel("Emotion")
axes4[0, 0].set_ylabel("Count")
axes4[0, 0].tick_params(axis="x", rotation=45)

ct_gender_comb = pd.crosstab(combined["emotion"], combined["gender"]).reindex(COMBINED_EMOTION_ORDER)
ct_gender_comb.plot(kind="bar", ax=axes4[0, 1], color=["#55A868", "#C44E52"])
axes4[0, 1].set_title("Emotion × Gender")
axes4[0, 1].set_xlabel("Emotion")
axes4[0, 1].set_ylabel("Count")
axes4[0, 1].tick_params(axis="x", rotation=45)

axes4[1, 0].hist(combined["duration_sec"], bins=30, color="#4C72B0", edgecolor="white")
axes4[1, 0].set_title("Audio Duration Distribution")
axes4[1, 0].set_xlabel("Duration (seconds)")
axes4[1, 0].set_ylabel("Count")

df_comb_plot = combined.copy()
df_comb_plot["emotion"] = pd.Categorical(df_comb_plot["emotion"], categories=COMBINED_EMOTION_ORDER, ordered=True)
df_comb_plot.boxplot(column="duration_sec", by="emotion", ax=axes4[1, 1])
axes4[1, 1].set_title("Duration by Emotion")
axes4[1, 1].set_xlabel("Emotion")
axes4[1, 1].set_ylabel("Duration (seconds)")
axes4[1, 1].tick_params(axis="x", rotation=45)
plt.sca(axes4[1, 1])
plt.title("Duration by Emotion")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "combined_eda_overview.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'combined_eda_overview.png')}")
plt.close()

fig5, axes5 = plt.subplots(1, 2, figsize=(20, 5))
fig5.suptitle("Combined Actor-Level Analysis (Actors 1–115)", fontsize=14, fontweight="bold")

actor_counts_comb = combined["actor_id"].value_counts().sort_index()
bar_colors = ["#4C72B0" if aid <= 24 else "#DD8452" for aid in actor_counts_comb.index]
axes5[0].bar(actor_counts_comb.index, actor_counts_comb.values, color=bar_colors)
axes5[0].set_title("Files per Actor  (blue = RAVDESS, orange = CREMA-D)")
axes5[0].set_xlabel("Actor ID")
axes5[0].set_ylabel("Count")
all_ids = sorted(combined["actor_id"].unique())
axes5[0].set_xticks([x for x in all_ids if x % 5 == 0 or x == all_ids[0]])
axes5[0].tick_params(axis="x", rotation=45, labelsize=8)

actor_dur_comb = combined.groupby("actor_id")["duration_sec"].mean().sort_index()
bar_colors_dur = ["#4C72B0" if aid <= 24 else "#DD8452" for aid in actor_dur_comb.index]
axes5[1].bar(actor_dur_comb.index, actor_dur_comb.values, color=bar_colors_dur)
axes5[1].set_title("Mean Duration per Actor  (blue = RAVDESS, orange = CREMA-D)")
axes5[1].set_xlabel("Actor ID")
axes5[1].set_ylabel("Duration (seconds)")
axes5[1].set_xticks([x for x in all_ids if x % 5 == 0 or x == all_ids[0]])
axes5[1].tick_params(axis="x", rotation=45, labelsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(OUTPUT_DIR, "combined_eda_actors.png"), dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.join(OUTPUT_DIR, 'combined_eda_actors.png')}")
plt.close()

combined["dataset"] = combined["actor_id"].apply(lambda x: "RAVDESS" if x <= 24 else "CREMA-D")

fig6, axes6 = plt.subplots(1, 2, figsize=(14, 5))
fig6.suptitle("Combined Dataset — Source Breakdown", fontsize=14, fontweight="bold")

ct_source = pd.crosstab(combined["emotion"], combined["dataset"]).reindex(COMBINED_EMOTION_ORDER)
ct_source.plot(kind="bar", ax=axes6[0], color=["#DD8452", "#4C72B0"])
axes6[0].set_title("Emotion Distribution by Dataset")
axes6[0].set_xlabel("Emotion")
axes6[0].set_ylabel("Count")
axes6[0].tick_params(axis="x", rotation=45)

ct_intensity_comb = pd.crosstab(combined["emotion"], combined["intensity"]).reindex(COMBINED_EMOTION_ORDER)
ct_intensity_comb.plot(kind="bar", ax=axes6[1], color=["#4C72B0", "#DD8452"])
axes6[1].set_title("Emotion × Intensity")
axes6[1].set_xlabel("Emotion")
axes6[1].set_ylabel("Count")
axes6[1].tick_params(axis="x", rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(OUTPUT_DIR, "combined_eda_sources.png"), dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.join(OUTPUT_DIR, 'combined_eda_sources.png')}")
plt.close()

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
