"""Network registry: feature extractors over the Dict observation (ADR-0002).

Each extractor encodes obs["market"] of shape (batch, num_pairs, window, num
features) plus obs["assets"] of shape (batch, num_pairs, 3) into a flat
feature vector consumed by the SB3 actor/critic heads. All extractors share
weights across pairs (the same encoder is applied to every pair).

Default hyperparameters follow common SB3 extractor sizing (features_dim 128,
small hidden layers); experiment YAMLs override them via network.kwargs.
"""

from __future__ import annotations

import gymnasium
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


def _market_shape(observation_space: gymnasium.spaces.Dict) -> tuple[int, int, int]:
    """Return (num_pairs, window_size, num_features) of the market box.

    Args:
        observation_space: Dict observation space of ForexEnv.

    Returns:
        Tuple of market tensor dimensions.
    """
    shape = observation_space["market"].shape
    return int(shape[0]), int(shape[1]), int(shape[2])


def _assets_dim(observation_space: gymnasium.spaces.Dict) -> int:
    """Return the flattened size of the assets box.

    Args:
        observation_space: Dict observation space of ForexEnv.

    Returns:
        Flattened assets dimension (num_pairs * 3).
    """
    shape = observation_space["assets"].shape
    return int(shape[0]) * int(shape[1])


class MlpExtractor(BaseFeaturesExtractor):
    """Baseline extractor: flatten everything and apply a two-layer MLP."""

    def __init__(
        self,
        observation_space: gymnasium.spaces.Dict,
        features_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize the extractor.

        Args:
            observation_space: Dict observation space of ForexEnv.
            features_dim: Output feature vector size.
            hidden_dim: Hidden layer width.
        """
        super().__init__(observation_space, features_dim)
        num_pairs, window, num_features = _market_shape(observation_space)
        input_dim = num_pairs * window * num_features + _assets_dim(observation_space)
        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observations into the feature vector.

        Args:
            observations: Dict of market/assets tensors.

        Returns:
            Tensor of shape (batch, features_dim).
        """
        flat = torch.cat(
            [observations["market"].flatten(1), observations["assets"].flatten(1)],
            dim=1,
        )
        return self._net(flat)


class Cnn1dExtractor(BaseFeaturesExtractor):
    """Temporal 1D-CNN over the window, shared across pairs."""

    def __init__(
        self,
        observation_space: gymnasium.spaces.Dict,
        features_dim: int = 128,
        channels: tuple[int, ...] = (32, 64),
        kernel_size: int = 5,
    ) -> None:
        """Initialize the extractor.

        Args:
            observation_space: Dict observation space of ForexEnv.
            features_dim: Output feature vector size.
            channels: Output channels of the successive stride-2 conv layers.
            kernel_size: Convolution kernel length along the time axis.
        """
        super().__init__(observation_space, features_dim)
        num_pairs, window, num_features = _market_shape(observation_space)
        self._num_pairs = num_pairs
        layers: list[nn.Module] = []
        in_channels = num_features
        for out_channels in channels:
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            in_channels = out_channels
        self._conv = nn.Sequential(*layers)
        with torch.no_grad():
            probe = torch.zeros(1, num_features, window)
            conv_out = int(self._conv(probe).flatten(1).shape[1])
        self._head = nn.Sequential(
            nn.Linear(
                num_pairs * conv_out + _assets_dim(observation_space), features_dim
            ),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observations into the feature vector.

        Args:
            observations: Dict of market/assets tensors.

        Returns:
            Tensor of shape (batch, features_dim).
        """
        market = observations["market"]
        batch = market.shape[0]
        # (B, N, W, F) -> (B*N, F, W): channels = features, length = window.
        stacked = market.reshape(-1, market.shape[2], market.shape[3]).permute(0, 2, 1)
        encoded = self._conv(stacked).flatten(1).reshape(batch, -1)
        flat = torch.cat([encoded, observations["assets"].flatten(1)], dim=1)
        return self._head(flat)


class LstmExtractor(BaseFeaturesExtractor):
    """LSTM over the window per pair; the last hidden state summarizes it.

    Note: this is a window encoder. For recurrence across env steps (memory
    beyond the window), use the `recurrent_ppo` algorithm instead.
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Dict,
        features_dim: int = 128,
        hidden_size: int = 64,
        num_layers: int = 1,
    ) -> None:
        """Initialize the extractor.

        Args:
            observation_space: Dict observation space of ForexEnv.
            features_dim: Output feature vector size.
            hidden_size: LSTM hidden state size.
            num_layers: Number of stacked LSTM layers.
        """
        super().__init__(observation_space, features_dim)
        num_pairs, _, num_features = _market_shape(observation_space)
        self._lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self._head = nn.Sequential(
            nn.Linear(
                num_pairs * hidden_size + _assets_dim(observation_space), features_dim
            ),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observations into the feature vector.

        Args:
            observations: Dict of market/assets tensors.

        Returns:
            Tensor of shape (batch, features_dim).
        """
        market = observations["market"]
        batch = market.shape[0]
        stacked = market.reshape(-1, market.shape[2], market.shape[3])  # (B*N, W, F)
        _, (hidden, _) = self._lstm(stacked)
        encoded = hidden[-1].reshape(batch, -1)
        flat = torch.cat([encoded, observations["assets"].flatten(1)], dim=1)
        return self._head(flat)


class AttentionExtractor(BaseFeaturesExtractor):
    """Small Transformer encoder over the window per pair."""

    def __init__(
        self,
        observation_space: gymnasium.spaces.Dict,
        features_dim: int = 128,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 64,
    ) -> None:
        """Initialize the extractor.

        Args:
            observation_space: Dict observation space of ForexEnv.
            features_dim: Output feature vector size.
            d_model: Token embedding size.
            num_heads: Attention heads (must divide d_model).
            num_layers: Encoder layers.
            feedforward_dim: Feedforward width inside the encoder.
        """
        super().__init__(observation_space, features_dim)
        num_pairs, window, num_features = _market_shape(observation_space)
        self._input_proj = nn.Linear(num_features, d_model)
        self._positional = nn.Parameter(torch.zeros(1, window, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._head = nn.Sequential(
            nn.Linear(
                num_pairs * d_model + _assets_dim(observation_space), features_dim
            ),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observations into the feature vector.

        Args:
            observations: Dict of market/assets tensors.

        Returns:
            Tensor of shape (batch, features_dim).
        """
        market = observations["market"]
        batch = market.shape[0]
        stacked = market.reshape(-1, market.shape[2], market.shape[3])  # (B*N, W, F)
        tokens = self._input_proj(stacked) + self._positional
        encoded = self._encoder(tokens)[:, -1, :].reshape(batch, -1)
        flat = torch.cat([encoded, observations["assets"].flatten(1)], dim=1)
        return self._head(flat)


class CrossPairAttentionExtractor(BaseFeaturesExtractor):
    """Self-attention ACROSS PAIRS (tokens = pairs), not across time.

    Each pair's window is encoded by a shared MLP into one token; a
    Transformer encoder then exchanges information between pairs, which is
    the natural inductive bias for cross-sectional (relative-value) tasks:
    the encoder is shared across pairs and the attention is permutation
    equivariant over them. Per-pair asset state is concatenated into each
    token before attention.
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Dict,
        features_dim: int = 128,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 128,
    ) -> None:
        """Initialize the extractor.

        Args:
            observation_space: Dict observation space of ForexEnv.
            features_dim: Output feature vector size.
            d_model: Per-pair token size.
            num_heads: Attention heads (must divide d_model).
            num_layers: Encoder layers.
            feedforward_dim: Feedforward width inside the encoder.
        """
        super().__init__(observation_space, features_dim)
        num_pairs, window, num_features = _market_shape(observation_space)
        assets_per_pair = int(observation_space["assets"].shape[1])
        self._token_encoder = nn.Sequential(
            nn.Linear(window * num_features + assets_per_pair, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._head = nn.Sequential(
            nn.Linear(num_pairs * d_model, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observations into the feature vector.

        Args:
            observations: Dict of market/assets tensors.

        Returns:
            Tensor of shape (batch, features_dim).
        """
        market = observations["market"]  # (B, N, W, F)
        assets = observations["assets"]  # (B, N, A)
        per_pair = torch.cat([market.flatten(2), assets], dim=2)  # (B, N, W*F+A)
        tokens = self._token_encoder(per_pair)  # (B, N, d_model)
        encoded = self._encoder(tokens)  # attention across the pair axis
        return self._head(encoded.flatten(1))


NETWORK_REGISTRY: dict[str, type[BaseFeaturesExtractor]] = {
    "mlp": MlpExtractor,
    "cnn1d": Cnn1dExtractor,
    "lstm": LstmExtractor,
    "attention": AttentionExtractor,
    "xattention": CrossPairAttentionExtractor,
}
