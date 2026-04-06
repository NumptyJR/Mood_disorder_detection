"""
CREMA-D + Combined Data Understanding Script
FileName: cremad_understanding
Authors: Joshua Schaff, Isaac Campbell, Cameron Bender
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD DATA
cremad_raw = pd.read_csv(os.path.join(OUTPUT_DIR, "cremad_metadata.csv"))
cremad = cremad_raw[cremad_raw["is_valid"] == True].copy()

ravdess = pd.read_csv(os.path.join(OUTPUT_DIR, "ravdess_metadata.csv"))
ravdess = ravdess[ravdess["is_valid"] == True].copy()

combined = pd.concat([ravdess, cremad], ignore_index=True)

CREMAD_EMOTION_ORDER = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
COMBINED_EMOTION_ORDER = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# CREMA-D
print("=" * 60)
print("CREMA-D DATA DESCRIPTION")
print("=" * 60)
print(f"Valid files:  {len(cremad)}")
print(f"Emotions:     {sorted(cremad['emotion'].unique())}")
print(f"Actors:       {cremad['actor_id'].nunique()} (IDs {cremad['actor_id'].min()}–{cremad['actor_id'].max()})")
print(f"Genders:      {dict(cremad['gender'].value_counts())}")
print(f"Intensities:  {dict(cremad['intensity'].value_counts())}")

print("\n--- Emotion counts ---")
print(cremad["emotion"].value_counts().sort_index().to_string())

print("\n--- Emotion × Gender ---")
print(pd.crosstab(cremad["emotion"], cremad["gender"]).to_string())

print("\n--- Emotion × Intensity ---")
print(pd.crosstab(cremad["emotion"], cremad["intensity"]).to_string())

print(f"\n--- Audio Properties ---")
print(f"Sample rates found: {cremad['sample_rate'].unique()}")
print(f"Duration (seconds): min={cremad['duration_sec'].min()}, "
      f"max={cremad['duration_sec'].max()}, "
      f"mean={cremad['duration_sec'].mean():.2f}, "
      f"std={cremad['duration_sec'].std():.2f}")

# CREMA-D EDA overview
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("CREMA-D Dataset — Exploratory Data Analysis", fontsize=16, fontweight="bold")

# Emotion distribution
emotion_counts = cremad["emotion"].value_counts().reindex(CREMAD_EMOTION_ORDER)
colors = sns.color_palette("husl", len(CREMAD_EMOTION_ORDER))
axes[0, 0].bar(CREMAD_EMOTION_ORDER, emotion_counts.values, color=colors)
axes[0, 0].set_title("Emotion Distribution")
axes[0, 0].set_xlabel("Emotion")
axes[0, 0].set_ylabel("Count")
axes[0, 0].tick_params(axis="x", rotation=45)

# Emotion by gender
ct_gender = pd.crosstab(cremad["emotion"], cremad["gender"]).reindex(CREMAD_EMOTION_ORDER)
ct_gender.plot(kind="bar", ax=axes[0, 1], color=["#55A868", "#C44E52"])
axes[0, 1].set_title("Emotion × Gender")
axes[0, 1].set_xlabel("Emotion")
axes[0, 1].set_ylabel("Count")
axes[0, 1].tick_params(axis="x", rotation=45)

# Duration distribution
axes[1, 0].hist(cremad["duration_sec"], bins=30, color="#4C72B0", edgecolor="white")
axes[1, 0].set_title("Audio Duration Distribution")
axes[1, 0].set_xlabel("Duration (seconds)")
axes[1, 0].set_ylabel("Count")

# Duration by emotion (box plot)
df_plot = cremad.copy()
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

# CREMA-D actor breakdown
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
fig2.suptitle("CREMA-D Actor-Level Analysis", fontsize=14, fontweight="bold")

actor_counts = cremad["actor_id"].value_counts().sort_index()
axes2[0].bar(actor_counts.index, actor_counts.values, color="#4C72B0")
axes2[0].set_title("Files per Actor")
axes2[0].set_xlabel("Actor ID")
axes2[0].set_ylabel("Count")
axes2[0].set_xticks(sorted(cremad["actor_id"].unique()))
axes2[0].tick_params(axis="x", rotation=90)

actor_dur = cremad.groupby("actor_id")["duration_sec"].mean().sort_index()
axes2[1].bar(actor_dur.index, actor_dur.values, color="#55A868")
axes2[1].set_title("Mean Duration per Actor")
axes2[1].set_xlabel("Actor ID")
axes2[1].set_ylabel("Duration (seconds)")
axes2[1].set_xticks(sorted(cremad["actor_id"].unique()))
axes2[1].tick_params(axis="x", rotation=90)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(OUTPUT_DIR, "cremad_eda_actors.png"), dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.join(OUTPUT_DIR, 'cremad_eda_actors.png')}")
plt.close()

# CREMA-D intensity breakdown
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle("CREMA-D Intensity Analysis", fontsize=14, fontweight="bold")

intensity_counts = cremad["intensity"].value_counts()
axes3[0].bar(intensity_counts.index, intensity_counts.values, color=["#4C72B0", "#DD8452"])
axes3[0].set_title("Overall Intensity Distribution")
axes3[0].set_xlabel("Intensity")
axes3[0].set_ylabel("Count")

ct_intensity = pd.crosstab(cremad["emotion"], cremad["intensity"]).reindex(CREMAD_EMOTION_ORDER)
ct_intensity.plot(kind="bar", ax=axes3[1], color=["#4C72B0", "#DD8452"])
axes3[1].set_title("Emotion × Intensity")
axes3[1].set_xlabel("Emotion")
axes3[1].set_ylabel("Count")
axes3[1].tick_params(axis="x", rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(OUTPUT_DIR, "cremad_eda_intensity.png"), dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.join(OUTPUT_DIR, 'cremad_eda_intensity.png')}")
plt.close()


# COMBINED
print("\n" + "=" * 60)
print("COMBINED DATA DESCRIPTION")
print("=" * 60)
print(f"Total files:  {len(combined)}")
print(f"  RAVDESS:    {len(ravdess)}")
print(f"  CREMA-D:    {len(cremad)}")
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

print(f"\n--- Audio Properties ---")
print(f"Sample rates found: {combined['sample_rate'].unique()}")
print(f"Duration (seconds): min={combined['duration_sec'].min()}, "
      f"max={combined['duration_sec'].max()}, "
      f"mean={combined['duration_sec'].mean():.2f}, "
      f"std={combined['duration_sec'].std():.2f}")

# Combined EDA overview
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

# Combined actor breakdown
fig5, axes5 = plt.subplots(1, 2, figsize=(20, 5))
fig5.suptitle("Combined Actor-Level Analysis (Actors 1–53)", fontsize=14, fontweight="bold")

actor_counts_comb = combined["actor_id"].value_counts().sort_index()
bar_colors = ["#4C72B0" if aid <= 24 else "#DD8452" for aid in actor_counts_comb.index]
axes5[0].bar(actor_counts_comb.index, actor_counts_comb.values, color=bar_colors)
axes5[0].set_title("Files per Actor  (blue = RAVDESS, orange = CREMA-D)")
axes5[0].set_xlabel("Actor ID")
axes5[0].set_ylabel("Count")
axes5[0].set_xticks(sorted(combined["actor_id"].unique()))
axes5[0].tick_params(axis="x", rotation=90)

actor_dur_comb = combined.groupby("actor_id")["duration_sec"].mean().sort_index()
bar_colors_dur = ["#4C72B0" if aid <= 24 else "#DD8452" for aid in actor_dur_comb.index]
axes5[1].bar(actor_dur_comb.index, actor_dur_comb.values, color=bar_colors_dur)
axes5[1].set_title("Mean Duration per Actor  (blue = RAVDESS, orange = CREMA-D)")
axes5[1].set_xlabel("Actor ID")
axes5[1].set_ylabel("Duration (seconds)")
axes5[1].set_xticks(sorted(combined["actor_id"].unique()))
axes5[1].tick_params(axis="x", rotation=90)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(os.path.join(OUTPUT_DIR, "combined_eda_actors.png"), dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.join(OUTPUT_DIR, 'combined_eda_actors.png')}")
plt.close()

# Combined dataset source breakdown
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
