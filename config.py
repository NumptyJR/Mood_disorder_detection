import torch

# --- Paths ---
METADATA_CSV = "eda_output/ravdess_metadata.csv"   # path to your combined CSV (song + speech)
AUDIO_ROOT   = ""                             # prepended to filepath column; set to "" if paths are absolute

# --- Audio ---
SAMPLE_RATE  = 16000      # resample everything to 16kHz (saves memory vs 48kHz)
CLIP_SECONDS = 5          # clips longer than this are truncated; shorter ones are padded
N_MELS       = 64         # mel filterbanks
N_FFT        = 1024       # FFT window size
HOP_LENGTH   = 512        # hop between frames (~32ms at 16kHz)
F_MIN        = 50         # minimum mel frequency (Hz)
F_MAX        = 8000       # maximum mel frequency (Hz)

# --- Emotions ---
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful"]
NUM_CLASSES = len(EMOTIONS)

# --- Split (by actor ID, not randomly) ---
# 24 actors total. Held out actors for val and test — never seen during training.
VAL_ACTORS  = [19, 20, 21, 22]   # ~16% of data
TEST_ACTORS = [23, 24]           # ~8% of data
# Remaining 18 actors → training

# --- Model ---
CNN_CHANNELS   = [32, 64, 128]   # feature maps per CNN block
LSTM_HIDDEN    = 128             # BiLSTM hidden size (x2 for bidirectional)
LSTM_LAYERS    = 2
DROPOUT        = 0.4

# --- Training ---
BATCH_SIZE     = 16              # smaller batches → more gradient updates, helps rare classes
NUM_EPOCHS     = 80              # more headroom since we have early stopping
LEARNING_RATE  = 1e-4            # lower LR for more stable convergence
WEIGHT_DECAY   = 1e-4
LR_PATIENCE    = 10              # more patience before reducing LR
LR_FACTOR      = 0.5
EARLY_STOP     = 20              # more patience before stopping

# --- Label smoothing ---
# Prevents the model from becoming overconfident on easy classes (neutral/angry)
# which crowds out harder ones (sad). 0.1 is a mild, safe value.
LABEL_SMOOTHING = 0.1

# --- SpecAugment ---
FREQ_MASK_MAX  = 15              # max mel bins to mask
TIME_MASK_MAX  = 30              # max time frames to mask
NUM_MASKS      = 2               # how many of each mask to apply

# --- Misc ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

SEED   = 42
CHECKPOINT_DIR = "checkpoints/"