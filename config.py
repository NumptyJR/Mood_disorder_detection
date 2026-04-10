import torch

#Paths
#METADATA_CSV = "data/ravdess_metadata.csv"   # path to your combined CSV (song + speech)
METADATA_CSV = "combined_metadata.csv"
AUDIO_ROOT   = ""

#Spectrogram cache
# Run buildCache.py once to pre-compute spectrograms.
# It will print the exact CACHE_DIR path to set here.
# Set to None to disable caching and compute spectrograms on the fly.
CACHE_DIR = "cache/mel_01765c90"


#Audio
SAMPLE_RATE  = 16000      # resample everything to 16kHz (saves memory vs 48kHz)
CLIP_SECONDS = 5          # clips longer than this are truncated; shorter ones are padded
N_MELS       = 64         # mel filterbanks
N_FFT        = 1024       # FFT window size
HOP_LENGTH   = 512        # hop between frames (~32ms at 16kHz)
F_MIN        = 50         # minimum mel frequency (Hz)
F_MAX        = 8000       # maximum mel frequency (Hz)

#Classification mode
# Set BINARY_MODE = True  for 2-class mood detector (neutral/positive vs negative)
# Set BINARY_MODE = False for 8-class emotion recogniser (full emotion labels)
BINARY_MODE = False

# Set USE_CLASS_WEIGHTS = True for inverse-frequency weights applied to loss
# Set USE_CLASS_WEIGHTS = False for all classes weighted equally
USE_CLASS_WEIGHTS = True

#Emotion classes
NEGATIVE_EMOTIONS = {"sad", "angry", "fearful", "disgust"}
POSITIVE_EMOTIONS = {"neutral", "calm", "happy", "surprised"}
ALL_EMOTIONS      = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
TRIMED_EMOTIONS      = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust"] # no surprised
CREMAD_EMOTIONS   = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]  # no calm/surprised

# Set DATASET = "combined" when using both RAVDESS + CREMAD
# Set DATASET = "cremad"   when using CREMAD only
DATASET = "combined"

if BINARY_MODE:
    EMOTIONS    = ["neutral/positive", "negative"]
    NUM_CLASSES = 2
    BATCH_SIZE      = 32
    LABEL_SMOOTHING = 0.05
    DROPOUT         = 0.4
    WEIGHT_DECAY    = 1e-4
else:
    EMOTIONS    = CREMAD_EMOTIONS if DATASET == "cremad" else ALL_EMOTIONS
    NUM_CLASSES = len(EMOTIONS)
    BATCH_SIZE      = 16
    LABEL_SMOOTHING = 0.1
    DROPOUT         = 0.5
    WEIGHT_DECAY    = 3e-4

# Checkpoint directory is mode-specific so the two models don't overwrite each other
CHECKPOINT_DIR = "checkpoints/binary/" if BINARY_MODE else "checkpoints/multiclass/"

# --- Split (by actor ID, not randomly) ---
# 24 actors total. Held out actors for val and test — never seen during training.
#VAL_ACTORS  = [19, 20, 21, 22]   # ~16% of data
#TEST_ACTORS = [23, 24]           # ~8% of data

# RAVDESS actors 1-24, CREMAD actors 25-115.
VAL_ACTORS  = [19, 20, 21, 22] + list(range(96, 106))   # 4 RAVDESS + 10 CREMAD → 66% CREMAD
TEST_ACTORS = [23, 24] + list(range(106, 116))           # 2 RAVDESS + 10 CREMAD → 80% CREMAD
# Remaining 18 actors for training

#Model
CNN_CHANNELS   = [32, 64, 128]
LSTM_HIDDEN    = 128
LSTM_LAYERS    = 2
# DROPOUT, BATCH_SIZE, LABEL_SMOOTHING, WEIGHT_DECAY set above by BINARY_MODE

#Training
NUM_EPOCHS     = 300             # high cap — early stopping will terminate before this
LEARNING_RATE  = 1e-4
LR_PATIENCE    = 25
LR_FACTOR      = 0.5
EARLY_STOP     = 50

#SpecAugment
FREQ_MASK_MAX  = 15              # max mel bins to mask
TIME_MASK_MAX  = 30              # max time frames to mask
NUM_MASKS      = 2               # how many of each mask to apply

#Misc
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

SEED   = 42