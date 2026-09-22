from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.3,
        activation: str = "ReLU",
        **kwargs,
    ):
        super(ClassificationHead, self).__init__()

        # Get configuration parameters with defaults
        hidden_dim = hidden_dim if hidden_dim is not None else input_dim // 2

        # Select activation function
        if activation == "ReLU":
            self.activation_fn = F.relu
        elif activation == "LeakyReLU":
            self.activation_fn = lambda t: F.leaky_relu(t, negative_slope=0.1)
        elif activation == "GELU":
            self.activation_fn = F.gelu
        else:
            self.activation_fn = F.relu  # Default fallback

        # Create a more robust architecture with batch normalization
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Add an additional layer for more capacity
        self.linear2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First layer
        x = self.linear1(x)

        # Handle batch norm for different input shapes
        if len(x.shape) == 3:  # (batch_size, seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = x.shape
            x = x.reshape(-1, hidden_dim)
            x = self.batch_norm1(x)
            x = x.reshape(batch_size, seq_len, hidden_dim)
        else:  # (batch_size, hidden_dim)
            x = self.batch_norm1(x)

        x = self.activation_fn(x)
        x = self.dropout1(x)

        # Second layer
        x = self.linear2(x)

        # Handle batch norm again
        if len(x.shape) == 3:  # (batch_size, seq_len, hidden_dim//2)
            batch_size, seq_len, hidden_dim = x.shape
            x = x.reshape(-1, hidden_dim)
            x = self.batch_norm2(x)
            x = x.reshape(batch_size, seq_len, hidden_dim)
        else:  # (batch_size, hidden_dim//2)
            x = self.batch_norm2(x)

        x = self.activation_fn(x)
        x = self.dropout2(x)

        # Output layer
        x = self.output_layer(x)

        return x


class RNNHead(nn.Module):
    """
    Flexible RNN head that can use either LSTM or GRU for temporal modeling.

    This head takes sequence embeddings from the TemporalFormer backbone and
    applies RNN-based temporal modeling to produce either sequence-level or
    episode-level predictions.

    Designed to be the second stage of the original TemporalFormer architecture,
    now separated for better modularity.
    """

    def __init__(
        self,
        input_dim: int,
        rnn_type: str = "GRU",
        rnn_layers: int = 1,
        rnn_hidden_mult: int = 4,
        num_classes: int = 4,
        dropout: float = 0.2,
        sequence_output: bool = True,
        bidirectional: bool = True,
        input_seq_len: int = 32,  # Patch sequence length from backbone
        target_seq_len: int = 384,  # Target temporal sequence length
        **kwargs,
    ):
        super(RNNHead, self).__init__()

        self.input_dim = input_dim
        self.rnn_type = rnn_type.upper()
        self.rnn_layers = rnn_layers
        self.rnn_hidden_size = input_dim * rnn_hidden_mult
        self.num_classes = num_classes
        self.dropout = dropout
        self.sequence_output = sequence_output
        self.bidirectional = bidirectional
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len

        # Validate upsampling parameters
        if sequence_output and target_seq_len % input_seq_len != 0:
            raise ValueError(
                f"target_seq_len ({target_seq_len}) must be divisible by "
                f"input_seq_len ({input_seq_len})"
            )

        # Calculate upsampling factor
        self.upsample_factor = target_seq_len // input_seq_len if sequence_output else 1

        # Debug print
        # print(
        #     f"RNNHead initialized: {input_seq_len} -> {target_seq_len} (factor: {self.upsample_factor})"
        # )

        # Calculate RNN output dimension
        rnn_output_dim = self.rnn_hidden_size
        if bidirectional:
            rnn_output_dim *= 2

        # Add input layer normalization for better gradient flow
        self.input_layer_norm = nn.LayerNorm(input_dim)

        # Create RNN layer
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.rnn_hidden_size,
                num_layers=rnn_layers,
                dropout=dropout if rnn_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=self.rnn_hidden_size,
                num_layers=rnn_layers,
                dropout=dropout if rnn_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            raise ValueError(
                f"Unsupported RNN type: {self.rnn_type}. Use 'LSTM' or 'GRU'"
            )

        # Add output layer normalization
        self.output_layer_norm = nn.LayerNorm(rnn_output_dim)

        # Output projection layers with residual connections
        if sequence_output:
            # For sequence output, predict per-timestep
            self.output_layers = nn.ModuleList([
                nn.Linear(rnn_output_dim, rnn_output_dim // 2),
                nn.LayerNorm(rnn_output_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(rnn_output_dim // 2, num_classes),
            ])
        else:
            # For single output, pool the sequence
            self.output_layers = nn.ModuleList([
                nn.Linear(rnn_output_dim, rnn_output_dim // 2),
                nn.LayerNorm(rnn_output_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(rnn_output_dim // 2, rnn_output_dim // 4),
                nn.LayerNorm(rnn_output_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(rnn_output_dim // 4, num_classes),
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RNN-based temporal modeling.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
               from TemporalFormer backbone

        Returns:
            If sequence_output=True: [batch_size, seq_len, num_classes]
            If sequence_output=False: [batch_size, num_classes]
        """
        # Apply input layer normalization
        x = self.input_layer_norm(x)
        
        # RNN processing
        rnn_output, _ = self.rnn(
            x
        )  # [B, seq_len, rnn_hidden_size * (2 if bidirectional else 1)]

        # Apply output layer normalization
        rnn_output = self.output_layer_norm(rnn_output)

        # Output generation
        if self.sequence_output:
            # Per-timestep predictions with manual layer processing
            output = rnn_output
            for i, layer in enumerate(self.output_layers):
                if i == 0:  # First linear layer
                    output = layer(output)
                elif isinstance(layer, nn.LayerNorm):
                    output = layer(output)
                elif isinstance(layer, nn.ReLU):
                    output = layer(output)
                elif isinstance(layer, nn.Dropout):
                    output = layer(output)
                else:  # Final linear layer
                    output = layer(output)

            # Upsample to target temporal resolution if needed
            if self.target_seq_len != self.input_seq_len:
                # print(
                #     f"RNNHead: Upsampling {output.shape} by factor {self.upsample_factor}"
                # )
                output = output.repeat_interleave(
                    self.upsample_factor, dim=1
                )  # [B, target_seq_len, num_classes]
                # print(f"RNNHead: After upsampling: {output.shape}")
        else:
            # Pool sequence and predict single output
            # Use mean pooling across sequence dimension
            pooled = rnn_output.mean(
                dim=1
            )  # [B, rnn_hidden_size * (2 if bidirectional else 1)]
            
            # Apply output layers manually
            output = pooled
            for layer in self.output_layers:
                output = layer(output)

        return output


class MAEDecoderHead(nn.Module):
    """
    MAE (Masked Autoencoder) decoder head for self-supervised pretraining.
    
    Reconstructs masked patches from encoder embeddings for pixel/patch-level supervision.
    Designed to work with Vision Transformer backbones that output patch embeddings.
    """
    
    def __init__(
        self,
        encoder_embed_dim: int,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        patch_size: int = 16,
        input_channels: int = 3,  # For accelerometer data: AccV, AccML, AccAP
        freq_bins: int = None,  # For spectral data: frequency bins
        freq_patch: int = None,  # When set, enables 2D patch reconstruction mode
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_patches: int = 256,  # Maximum number of patches to support
        **kwargs,
    ):
        super(MAEDecoderHead, self).__init__()

        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.freq_bins = freq_bins
        self.freq_patch = freq_patch  # If set: 2D patch reconstruction mode

        # Decoder embedding - project from encoder dimension to decoder dimension
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)

        # Mask token - learnable parameter for masked patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Position embedding for decoder — lazily resized on first forward call.
        # max_patches is just an initial allocation; the buffer grows automatically
        # to fit the actual token count (block_len // patch_size or nH*nW).
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, max_patches, decoder_embed_dim), requires_grad=False
        )
        self._pos_embed_initialized = False  # flag for lazy init

        # Decoder transformer blocks
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder_blocks = nn.TransformerEncoder(
            decoder_layer, num_layers=decoder_depth
        )

        # Decoder norm
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Decoder prediction head - reconstruct full spectral shape
        if freq_patch is not None:
            # 2D patch mode (SpectralPatchEncoder): each token covers freq_patch × patch_size region
            patch_spectral_dim = input_channels * freq_patch * patch_size
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_spectral_dim)
            self.output_shape = (input_channels, freq_patch, patch_size)
        elif freq_bins is not None:
            # Temporal patch mode: each token covers all freq_bins × patch_size (1D temporal)
            patch_spectral_dim = input_channels * freq_bins * patch_size
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_spectral_dim)
            self.output_shape = (input_channels, freq_bins, patch_size)  # Store for reshape
        else:
            # For image data: square patches
            # Each patch has patch_size * patch_size * input_channels values
            patch_dim = patch_size * patch_size * input_channels
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim)
            self.output_shape = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights following MAE paper."""
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize position embeddings
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor = None,
        mask: torch.Tensor = None,
        grid_shape: tuple = None,
    ) -> torch.Tensor:
        """
        Forward pass for MAE decoder.

        Args:
            x: Encoder output [batch_size, num_patches, encoder_embed_dim]
            ids_restore: Optional patch-order restoration indices (unused in our MAE variant)
            mask: Optional mask tensor (returned unchanged if provided)
            grid_shape: (nH, nW) for 2D patch backbones (e.g. SpectralPatchEncoder).
                        When provided and freq_patch is set, reconstructs [B, C, nH*fp, nW*tp].

        Returns:
            Reconstructed signal [batch_size, channels, freq, time] (spectral) or
            [batch_size, channels, patch_dim] (image)
        """
        batch_size, num_visible, _ = x.shape
        
        # Embed tokens to decoder dimension
        x = self.decoder_embed(x)  # [B, num_visible, decoder_embed_dim]
        
        if ids_restore is not None:
            # Add mask tokens for missing patches
            num_patches = ids_restore.shape[1]
            mask_tokens = self.mask_token.repeat(
                batch_size, num_patches - num_visible, 1
            )
            
            # Concatenate visible and mask tokens
            x_full = torch.cat([x, mask_tokens], dim=1)  # [B, num_patches, decoder_embed_dim]
            
            # Unshuffle - restore original patch order
            x_full = torch.gather(
                x_full, dim=1, 
                index=ids_restore.unsqueeze(-1).repeat(1, 1, self.decoder_embed_dim)
            )
        else:
            # No masking, use all patches
            x_full = x
            num_patches = num_visible
        
        # Add position embeddings — auto-resize buffer to exact token count on first call.
        if not self._pos_embed_initialized or num_patches > self.decoder_pos_embed.shape[1]:
            new_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.decoder_embed_dim,
                            device=x.device, dtype=torch.float32),
                requires_grad=False,
            )
            torch.nn.init.normal_(new_embed, std=0.02)
            self.decoder_pos_embed = new_embed
            self._pos_embed_initialized = True
        pos_embed = self.decoder_pos_embed[:, :num_patches, :]
        
        x_full = x_full + pos_embed
        
        # Apply decoder transformer in float32 to prevent fp16 overflow → NaN in softmax
        x_full = self.decoder_blocks(x_full.float()).to(x_full.dtype)
        x_full = self.decoder_norm(x_full)
        
        # Predict values
        pred = self.decoder_pred(x_full)  # [B, num_patches, spectral_dim or patch_dim]
        
        # Reconstruct spatial layout from token predictions
        if self.freq_patch is not None and grid_shape is not None:
            # 2D patch mode (SpectralPatchEncoder)
            # pred: [B, nH*nW, C*freq_patch*time_patch]
            nH, nW = grid_shape
            batch_size = pred.shape[0]
            channels, fp, tp = self.output_shape  # (C, freq_patch, time_patch)
            pred = pred.view(batch_size, nH, nW, channels, fp, tp)
            pred = pred.permute(0, 3, 1, 4, 2, 5)  # [B, C, nH, fp, nW, tp]
            pred = pred.reshape(batch_size, channels, nH * fp, nW * tp)
            pred = pred.contiguous()
        elif self.freq_bins is not None:
            # Temporal patch mode: each token covers all freq_bins × patch_size
            batch_size, num_patches, patch_spectral_dim = pred.shape
            channels, freq_bins, patch_size = self.output_shape

            # Reshape each patch: [B, num_patches, C*H*patch_size] -> [B, num_patches, C, H, patch_size]
            pred = pred.view(batch_size, num_patches, channels, freq_bins, patch_size)

            # Rearrange to: [B, C, H, num_patches, patch_size]
            pred = pred.permute(0, 2, 3, 1, 4)

            # Flatten temporal patches to get full time sequence: [B, C, H, time_steps]
            time_steps = num_patches * patch_size
            pred = pred.contiguous().view(batch_size, channels, freq_bins, time_steps)
            pred = pred.contiguous()
        
        # Return predictions and mask if provided
        if mask is not None:
            return pred, mask
        return pred


class LinearProbeHead(nn.Module):
    """True linear probe: global average pool over sequence then a single linear layer.

    Takes [B, T, D] or [B, D] input. For sequence input, averages over T before
    the linear classifier — no temporal modeling capacity, purely tests backbone
    representation quality.
    """

    def __init__(self, input_dim: int, num_classes: int = 2, **kwargs):
        super(LinearProbeHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:          # [B, T, D] → mean over T → [B, D]
            x = x.mean(dim=1)
        return self.fc(x)         # [B, num_classes]


class IdentityHead(nn.Module):
    def __init__(self, **kwargs):
        super(IdentityHead, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class JEPAPredictorHead(nn.Module):
    """
    JEPA predictor head: lightweight transformer that predicts target embeddings
    for masked positions given visible embeddings.

    Takes visible patch embeddings from the student encoder, inserts learnable
    mask tokens at masked positions, adds positional embeddings, runs through
    a shallow transformer, and outputs predictions for masked positions only.
    """

    def __init__(
        self,
        encoder_embed_dim: int = 512,
        predictor_embed_dim: int = 256,
        predictor_depth: int = 2,
        num_heads: int = 8,
        max_patches: int = 256,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.predictor_embed_dim = predictor_embed_dim

        # Project from encoder dim to predictor dim
        self.encoder_to_predictor = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional embeddings for full sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, predictor_embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        layer = nn.TransformerEncoderLayer(
            d_model=predictor_embed_dim,
            nhead=num_heads,
            dim_feedforward=predictor_embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=predictor_depth)
        self.norm = nn.LayerNorm(predictor_embed_dim)

        # Project back to encoder dim
        self.predictor_to_encoder = nn.Linear(predictor_embed_dim, encoder_embed_dim)

    def forward(
        self,
        visible_embeddings: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            visible_embeddings: [B, num_visible, encoder_dim] from student encoder
            patch_mask: [B, num_patches] bool, True = masked

        Returns:
            predicted: [B, num_masked, encoder_dim] predictions for masked positions
        """
        B, num_visible, _ = visible_embeddings.shape
        num_patches = patch_mask.shape[1]
        num_masked = patch_mask.sum(dim=1)[0].int().item()  # Assumes uniform masking
        device = visible_embeddings.device

        # 1. Project visible embeddings to predictor dim
        visible_pred = self.encoder_to_predictor(visible_embeddings)  # [B, vis, pred_dim]
        dtype = visible_pred.dtype

        # 2. Build full sequence: place visible tokens and mask tokens
        visible_mask = ~patch_mask  # True = visible
        full_seq = torch.zeros(B, num_patches, self.predictor_embed_dim, device=device, dtype=dtype)

        # Place visible embeddings at their original positions
        full_seq[visible_mask] = visible_pred.reshape(-1, self.predictor_embed_dim)

        # Place mask tokens at masked positions (cast to match AMP dtype)
        mask_tokens = self.mask_token.to(dtype).expand(B, num_masked, -1)
        full_seq[patch_mask] = mask_tokens.reshape(-1, self.predictor_embed_dim)

        # 3. Add positional embeddings (cast to match AMP dtype)
        pos_embed = self.pos_embed[:, :num_patches, :].to(dtype)
        full_seq = full_seq + pos_embed

        # 4. Transformer blocks
        full_seq = self.blocks(full_seq)
        full_seq = self.norm(full_seq)

        # 5. Extract masked positions only
        masked_pred = full_seq[patch_mask].reshape(B, num_masked, self.predictor_embed_dim)

        # 6. Project back to encoder dim
        predicted = self.predictor_to_encoder(masked_pred)  # [B, num_masked, encoder_dim]

        return predicted


class SimCLRHead(nn.Module):
    """
    SimCLR projection head for contrastive learning.
    
    Projects encoder features to a lower dimensional space for contrastive loss computation.
    Typically consists of a 2-layer MLP with ReLU activation.
    """
    
    def __init__(
        self,
        encoder_dim: int = 768,
        projection_dim: int = 256,
        hidden_dim: int = None,
        temperature: float = 0.07,
        pool_patches: bool = False,
        pool_method: str = "mean",
        apply_per_patch: bool = False,
        **kwargs
    ):
        """
        Initialize SimCLR projection head.

        Args:
            encoder_dim: Dimension of encoder features
            projection_dim: Dimension of projected features
            hidden_dim: Hidden dimension for MLP (defaults to encoder_dim)
            temperature: Temperature for contrastive loss (not used in forward but stored for reference)
            pool_patches: If True, pool over patch dimension before projection.
                When backbone outputs [B, num_patches, embed_dim], pooling produces [B, embed_dim]
                instead of flattening to [B, num_patches * embed_dim]. Set encoder_dim to
                embed_dim (e.g. 512) when using this option.
            pool_method: Pooling operation when pool_patches=True. Options:
                "mean" - average over patches (default; can collapse to global statistics)
                "max"  - max over patches; captures peak activations, more selective than mean
            apply_per_patch: If True, apply MLP independently to each patch token and return
                [B, N, projection_dim]. The MLP is shared across positions (no flattening).
                Use with patch-level contrastive loss. Takes precedence over pool_patches.
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.pool_patches = pool_patches
        self.pool_method = pool_method
        self.apply_per_patch = apply_per_patch

        if hidden_dim is None:
            hidden_dim = encoder_dim

        # 2-layer MLP projection head as used in SimCLR paper
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection head.

        Args:
            x: Encoder features [batch_size, encoder_dim] or [batch_size, num_patches, embed_dim]

        Returns:
            Projected features:
              - [B, projection_dim]   when pool_patches=True or input is 2D
              - [B, N, projection_dim] when apply_per_patch=True
              - [B, N*D]              when pool_patches=False and apply_per_patch=False (flat)
        """
        if len(x.shape) > 2:
            if self.apply_per_patch:
                # Apply MLP to each token independently: [B, N, D] → [B, N, proj_dim]
                B, N, D = x.shape
                x = self.projection_head(x.reshape(B * N, D))
                return x.reshape(B, N, self.projection_dim)
            elif self.pool_patches:
                if self.pool_method == "max":
                    x = x.max(dim=1).values  # [B, N, D] → [B, D]
                else:
                    x = x.mean(dim=1)        # [B, N, D] → [B, D]
            else:
                x = x.view(x.size(0), -1)  # [B, N, D] → [B, N*D]

        return self.projection_head(x)


class SegmentationHead(nn.Module):
    """Per-timestep classification head for sequence segmentation tasks.

    Takes [B, T, input_dim] backbone output and produces per-step class logits
    [B, T, num_classes] via a small MLP.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, input_dim] → [B, T, num_classes]"""
        return self.mlp(x)
