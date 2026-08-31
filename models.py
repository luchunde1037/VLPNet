"""Neural network architecture for variable-length P-wave regression."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class VLPNet(nn.Module):
    """Predict the logarithmic displacement target from a variable-length waveform."""

    def __init__(
        self,
        in_channels=1,
        feature_channels=(32, 64),
        recurrent_hidden=64,
        recurrent_layers=2,
        dropout=0.3,
    ):
        super().__init__()
        channel_1, channel_2 = feature_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, channel_1, kernel_size=5, padding=2),
            nn.BatchNorm1d(channel_1),
            nn.ReLU(),
            nn.Conv1d(channel_1, channel_2, kernel_size=5, padding=2),
            nn.BatchNorm1d(channel_2),
            nn.ReLU(),
        )
        self.sequence_encoder = nn.LSTM(
            input_size=channel_2,
            hidden_size=recurrent_hidden,
            num_layers=recurrent_layers,
            batch_first=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(recurrent_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, waveforms, lengths):
        """
        Parameters
        ----------
        waveforms : torch.Tensor
            Padded waveform tensor with shape ``(batch, time, channels)``.
        lengths : torch.Tensor
            Number of valid samples in each waveform.
        """
        features = self.feature_extractor(waveforms.transpose(1, 2))
        features = features.transpose(1, 2)
        packed = pack_padded_sequence(
            features,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.sequence_encoder(packed)
        return self.regression_head(hidden[-1]).squeeze(-1)

