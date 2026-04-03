import torch
import torch.nn as nn
from config import *


class ConvBlock(nn.Module):
    """
    Conv2d → BatchNorm → ReLU → MaxPool → Dropout
    The CNN backbone treats the mel spectrogram like a 2D image,
    learning local frequency-time acoustic patterns.
    """

    def __init__(self, in_channels: int, out_channels: int, pool_size=(2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size),
            nn.Dropout2d(p=0.2),
        )

    def forward(self, x):
        return self.block(x)


class AttentionPooling(nn.Module):
    """
    Soft attention over the time axis of the LSTM output.
    Lets the model learn which time steps are most emotionally salient
    rather than blindly averaging across the whole sequence.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, time, hidden)
        scores = self.attention(lstm_out)  # (batch, time, 1)
        weights = torch.softmax(scores, dim=1)  # (batch, time, 1)
        pooled = (lstm_out * weights).sum(dim=1)  # (batch, hidden)
        return pooled


class SERModel(nn.Module):
    """
    Speech Emotion Recognition model trained from scratch.

    Architecture:
      Input: Log-Mel spectrogram  (batch, 1, n_mels, time_frames)
        ↓
      3 × ConvBlock               (learns local acoustic features)
        ↓
      Reshape → (batch, time', freq_channels)
        ↓
      2-layer Bidirectional LSTM  (captures temporal dynamics)
        ↓
      Attention pooling           (focus on emotionally rich frames)
        ↓
      Fully-connected head → softmax over NUM_CLASSES
    """

    def __init__(self):
        super().__init__()

        channels = CNN_CHANNELS  # e.g. [32, 64, 128]

        # CNN backbone
        self.cnn = nn.Sequential(
            ConvBlock(1, channels[0], pool_size=(2, 2)),
            ConvBlock(channels[0], channels[1], pool_size=(2, 2)),
            ConvBlock(channels[1], channels[2], pool_size=(2, 1)),
            # Final pool: (2,1) preserves more time resolution for the LSTM
        )

        # After 3 pooling ops on N_MELS=64:
        #   freq dim: 64 → 32 → 16 → 8
        # LSTM input size = freq_out * last_channel_count
        freq_out = N_MELS // (2 * 2 * 2)  # = 8
        lstm_input_size = freq_out * channels[2]  # = 8 * 128 = 1024

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )

        lstm_output_size = LSTM_HIDDEN * 2  # bidirectional

        self.attention = AttentionPooling(lstm_output_size)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_mels, time)
        x = self.cnn(x)  # (batch, C, freq', time')

        batch, C, freq, time = x.shape
        # Rearrange for LSTM: treat each time step as a feature vector
        x = x.permute(0, 3, 1, 2)  # (batch, time, C, freq)
        x = x.reshape(batch, time, C * freq)  # (batch, time, C*freq)

        x, _ = self.lstm(x)  # (batch, time, lstm_out)
        x = self.attention(x)  # (batch, lstm_out)
        x = self.classifier(x)  # (batch, num_classes)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SERModel()
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Smoke test with a dummy batch
    dummy = torch.randn(4, 1, N_MELS, 157)  # ~5s clip at 16kHz / HOP_LENGTH
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # should be (4, 6)