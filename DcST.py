import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SingleHeadSelfAttention(nn.Module):
    """Single-Head Self-Attention mechanism."""

    def __init__(self, input_dim, compression_rate=4, num_regions=4):
        super(SingleHeadSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.compression_rate = compression_rate
        self.compressed_dim = input_dim // compression_rate
        self.num_regions = num_regions

        # Compression of input dimensions
        self.conv_compress = nn.Conv2d(input_dim, self.compressed_dim, kernel_size=1)

        # Multi-scale pooling layers
        self.pool_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d((i, i)) for i in range(2, 2 + num_regions)
        ])

        # Inter-region information exchange
        self.dsc_layers = nn.ModuleList([
            nn.Conv2d(self.compressed_dim, self.compressed_dim, kernel_size=3,
                      padding=1, groups=self.compressed_dim)
            for _ in range(num_regions)
        ])

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.compressed_dim)

        # Query, key, value projections
        self.W_Q = nn.Linear(self.compressed_dim, self.compressed_dim)
        self.W_K = nn.Linear(self.compressed_dim, self.compressed_dim)
        self.W_V = nn.Linear(self.compressed_dim, self.compressed_dim)

        # Final projection back to original dimensions
        self.conv_restore = nn.Conv2d(self.compressed_dim, input_dim, kernel_size=1)

    def forward(self, x):
        batch_size, c, h, w = x.size()

        # Compress input dimensions
        x_compressed = self.conv_compress(x)  # B x (c/s) x h x w

        # Multi-scale pooling and inter-region information exchange
        pooled_features = []
        for i, (pool, dsc) in enumerate(zip(self.pool_layers, self.dsc_layers)):
            p_star = pool(x_compressed)  # B x (c/s) x i x i
            p = p_star + dsc(p_star)  # Apply DSC and residual
            pooled_features.append(p.reshape(batch_size, self.compressed_dim, -1))

        # Concatenate and apply layer norm
        P = torch.cat(pooled_features, dim=2).transpose(1, 2)  # B x m x (c/s)
        P = self.layer_norm(P)

        # Reshape x_compressed for query computation
        Q = self.W_Q(x_compressed.reshape(batch_size, self.compressed_dim, -1).transpose(1, 2))  # B x (h*w) x (c/s)
        K = self.W_K(P)  # B x m x (c/s)
        V = self.W_V(P)  # B x m x (c/s)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.compressed_dim)  # B x (h*w) x m
        attention = F.softmax(scores, dim=-1)  # B x (h*w) x m

        # Apply attention to values
        x_weighted_star = torch.matmul(attention, V)  # B x (h*w) x (c/s)

        # Reshape and project back to original dimensions
        x_weighted_star = x_weighted_star.transpose(1, 2).reshape(batch_size, self.compressed_dim, h, w)
        x_weighted = self.conv_restore(x_weighted_star)  # B x c x h x w

        return x_weighted


class MLP(nn.Module):
    """Multi-scale feature extraction module."""

    def __init__(self, input_dim, hidden_dim=None):
        super(MLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 4

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Input shape: B x C x H x W
        orig_shape = x.shape
        x = x.flatten(2).transpose(1, 2)  # B x (H*W) x C

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        # Restore original shape
        x = x.transpose(1, 2).reshape(orig_shape)
        return x


class SHformerBlock(nn.Module):
    """Single-Head Transformer Block."""

    def __init__(self, dim, compression_rate=4, num_regions=4):
        super(SHformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.shsa = SingleHeadSelfAttention(dim, compression_rate, num_regions)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        # Apply SHSA with residual connection and layer norm
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_att = self.shsa(x_norm) + x

        # Apply MLP with residual connection and layer norm
        x_norm2 = self.norm2(x_att.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_ffn = self.mlp(x_norm2) + x_att

        return x_ffn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_seq_length=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Input shape: B x L x D
        return x + self.pe[:, :x.size(1), :]


class DualChannelEncoder(nn.Module):
    """Dual-channel encoder with feature and temporal paths."""

    def __init__(self, input_dim, window_size, model_dim=128, num_blocks=3):
        super(DualChannelEncoder, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.model_dim = model_dim

        # Feature Channel
        self.feature_embed = nn.Linear(window_size, model_dim)
        self.feature_blocks = nn.ModuleList([
            SHformerBlock(model_dim) for _ in range(num_blocks)
        ])

        # Temporal Channel
        self.temporal_embed = nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, max_seq_length=window_size)
        self.temporal_blocks = nn.ModuleList([
            SHformerBlock(model_dim) for _ in range(num_blocks)
        ])

    def forward(self, x):
        # Input shape: B x W x D (batch, window_size, input_dim)
        batch_size = x.size(0)

        # Feature Channel (focus on feature relationships)
        # Transpose to B x D x W and project to B x D x model_dim
        x_feature = x.transpose(1, 2)  # B x D x W
        x_feature = self.feature_embed(x_feature)  # B x D x model_dim

        # Reshape for 2D processing in SHformer (treating as image-like data)
        x_feature = x_feature.unsqueeze(-1)  # B x D x model_dim x 1

        # Apply feature channel blocks
        for block in self.feature_blocks:
            x_feature = block(x_feature)

        # Reshape back
        x_feature = x_feature.squeeze(-1)  # B x D x model_dim

        # Temporal Channel (focus on temporal dependencies)
        x_temporal = self.temporal_embed(x)  # B x W x model_dim
        x_temporal = self.pos_encoding(x_temporal)  # Add positional encoding

        # Reshape for 2D processing
        x_temporal = x_temporal.unsqueeze(-1).transpose(1, 2)  # B x model_dim x W x 1

        # Apply temporal channel blocks
        for block in self.temporal_blocks:
            x_temporal = block(x_temporal)

        # Reshape back
        x_temporal = x_temporal.squeeze(-1).transpose(1, 2)  # B x W x model_dim

        return x_feature, x_temporal


class WeightedFusionGate(nn.Module):
    """Weighted Fusion Gate for prediction module."""

    def __init__(self, model_dim, input_dim):
        super(WeightedFusionGate, self).__init__()
        self.linear = nn.Linear(model_dim * 2, 2)  # For calculating weights
        self.fc = nn.Linear(model_dim * 2, model_dim)  # For prediction
        self.output_proj = nn.Linear(model_dim, input_dim)

    def forward(self, feature_embed, temporal_embed):
        # feature_embed: B x D x model_dim
        # temporal_embed: B x W x model_dim

        # Get the last time step from temporal embedding
        temporal_last = temporal_embed[:, -1, :]  # B x model_dim

        # Avg pool across features
        feature_avg = feature_embed.mean(dim=1)  # B x model_dim

        # Concatenate
        concat = torch.cat([feature_avg, temporal_last], dim=1)  # B x (2*model_dim)

        # Calculate weights
        weights = F.softmax(self.linear(concat), dim=1)  # B x 2

        # Apply weights
        weighted_concat = torch.cat([
            feature_avg * weights[:, 0].unsqueeze(1),
            temporal_last * weights[:, 1].unsqueeze(1)
        ], dim=1)  # B x (2*model_dim)

        # Process through FC
        output = self.fc(weighted_concat)  # B x model_dim

        # Project to prediction
        prediction = self.output_proj(output)  # B x input_dim

        return prediction


class SHformerDecoder(nn.Module):
    """Decoder based on SHformer for reconstruction."""

    def __init__(self, model_dim, window_size, input_dim, num_blocks=3):
        super(SHformerDecoder, self).__init__()
        self.model_dim = model_dim
        self.window_size = window_size
        self.input_dim = input_dim

        # Transformation layers
        self.temporal_transform = nn.Linear(model_dim, window_size * model_dim)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            SHformerBlock(model_dim) for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Linear(model_dim, input_dim)

    def forward(self, temporal_embed):
        # temporal_embed: B x W x model_dim
        batch_size = temporal_embed.size(0)

        # Transform to decoder space
        x = self.temporal_transform(temporal_embed.mean(dim=1))  # B x (W*model_dim)
        x = x.reshape(batch_size, self.window_size, self.model_dim)  # B x W x model_dim

        # Reshape for 2D processing
        x = x.transpose(1, 2).unsqueeze(-1)  # B x model_dim x W x 1

        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)

        # Reshape back
        x = x.squeeze(-1).transpose(1, 2)  # B x W x model_dim

        # Project to output
        output = self.output_proj(x)  # B x W x input_dim

        return output


class DualChannelSingleheadTransformer(nn.Module):
    """Complete DcST model with prediction and reconstruction components."""

    def __init__(self, input_dim, window_size, model_dim=128, num_blocks=3):
        super(DualChannelSingleheadTransformer, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.model_dim = model_dim

        # Dual-channel encoder
        self.encoder = DualChannelEncoder(input_dim, window_size, model_dim, num_blocks)

        # Prediction module (WFG)
        self.prediction_module = WeightedFusionGate(model_dim, input_dim)

        # Reconstruction module
        self.reconstruction_module = SHformerDecoder(model_dim, window_size, input_dim, num_blocks)

    def forward(self, x):
        # x: B x W x D (batch, window_size, input_dim)

        # Encode input using dual channels
        feature_embed, temporal_embed = self.encoder(x)

        # Prediction (next value)
        prediction = self.prediction_module(feature_embed, temporal_embed)

        # Reconstruction (original input)
        reconstruction = self.reconstruction_module(temporal_embed)

        return prediction, reconstruction

    def compute_anomaly_score(self, x, gamma=1.0):
        """
        Compute anomaly score based on prediction and reconstruction errors.

        Args:
            x: Input time series of shape B x W x D
            gamma: Balance coefficient for prediction and reconstruction errors

        Returns:
            Anomaly scores of shape B
        """
        # Get predictions and reconstructions
        prediction, reconstruction = self.forward(x)

        # Extract the actual next value for prediction comparison
        actual_next = x[:, -1, :]  # B x D

        # Compute prediction error (RMSE)
        pred_error = torch.sqrt(torch.mean((prediction - actual_next) ** 2, dim=1))  # B

        # Compute reconstruction error (RMSE)
        recon_error = torch.sqrt(torch.mean((reconstruction - x) ** 2, dim=(1, 2)))  # B

        anomaly_score = (pred_error + gamma * recon_error) / (1 + gamma)

        return anomaly_score