import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class BiLSTMSwaraClassifier(nn.Module):
    """
    Bi-directional LSTM for Swara Classification
    
    Think of this as an AI that reads music in both directions:
    - Forward: Sa → Re → Ga → Ma...
    - Backward: Ma → Ga → Re → Sa...
    
    This helps it understand musical context better.
    """
    
    def __init__(self, config):
        super(BiLSTMSwaraClassifier, self).__init__()
        self.config = config
        
        # Model parameters
        self.input_size = 1  # Pitch value (1 number per frame)
        self.hidden_size = config.HIDDEN_SIZE
        self.num_layers = config.NUM_LAYERS
        self.num_classes = len(config.BHAIRAVI_SWARAS) + 1  # +1 for PAD token
        self.dropout_rate = config.DROPOUT
        
        # === MODEL LAYERS ===
        
        # 1. Input processing layer
        # Converts raw pitch to richer representation
        self.input_projection = nn.Linear(self.input_size, self.hidden_size)
        
        # 2. Bi-directional LSTM layers
        # The core of the model - learns musical patterns
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,  # Input shape: (batch, sequence, features)
            bidirectional=True,  # Read music forwards and backwards
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # 3. Attention mechanism (optional enhancement)
        # Helps model focus on important parts of the melody
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size * 2,  # *2 because bidirectional
            num_heads=8,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # 4. Classification layers
        # Converts LSTM output to swara predictions
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size * 2, self.hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(self.dropout_rate / 2),
            nn.Linear(self.hidden_size, self.num_classes)
        )
        
        # 5. Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using best practices"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-hidden weights  
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Biases
                param.data.fill_(0.)
                # Set forget gate bias to 1 (helps with training)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
    
    def forward(self, x: torch.Tensor, use_attention: bool = True) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, 1)
            use_attention: Whether to use attention mechanism
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, num_classes)
        """
        batch_size, seq_length, _ = x.shape
        
        # Step 1: Project input to hidden dimension
        # Raw pitch → Rich representation
        x_proj = self.input_projection(x)  # (batch, seq, hidden_size)
        
        # Step 2: Pass through LSTM
        # Learn temporal patterns in both directions
        lstm_out, (hidden, cell) = self.lstm(x_proj)
        # lstm_out shape: (batch, seq, hidden_size * 2)
        
        # Step 3: Optional attention mechanism
        if use_attention and self.training:
            # Let model focus on important melody parts
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Residual connection
            lstm_out = lstm_out + attn_out
        
        # Step 4: Classify each time step
        # Convert LSTM features → Swara probabilities
        output = self.classifier(lstm_out)
        # output shape: (batch, seq, num_classes)
        
        return output
    
    def get_model_info(self) -> dict:
        """Returns information about the model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'Bi-LSTM',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate
        }

class TCNSwaraClassifier(nn.Module):
    """
    Temporal Convolutional Network (TCN) for Swara Classification
    
    Think of this as an AI that scans music with different sized "windows":
    - Small windows catch quick ornaments (gamakas)
    - Large windows catch long phrase patterns
    
    It's often faster than LSTM and good at capturing rhythmic patterns.
    """
    
    def __init__(self, config):
        super(TCNSwaraClassifier, self).__init__()
        self.config = config
        
        # Model parameters
        self.input_size = 1
        self.hidden_size = config.HIDDEN_SIZE
        self.num_classes = len(config.BHAIRAVI_SWARAS) + 1  # +1 for PAD
        self.dropout_rate = config.DROPOUT
        
        # TCN specific parameters
        self.num_levels = 6  # Number of dilation levels
        self.kernel_size = 3  # Convolution window size
        
        # === MODEL LAYERS ===
        
        # 1. Input projection
        self.input_projection = nn.Conv1d(
            in_channels=self.input_size,
            out_channels=self.hidden_size,
            kernel_size=1
        )
        
        # 2. TCN blocks with increasing dilation
        # Each block looks at wider time spans
        self.tcn_blocks = nn.ModuleList()
        
        for level in range(self.num_levels):
            dilation = 2 ** level  # Exponential dilation: 1, 2, 4, 8, 16, 32
            
            block = TCNBlock(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=self.kernel_size,
                dilation=dilation,
                dropout=self.dropout_rate
            )
            self.tcn_blocks.append(block)
        
        # 3. Global feature aggregation
        # Combine information from all time scales
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_mixer = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate / 2),
            nn.Linear(self.hidden_size // 2, self.num_classes)
        )
        
        # 5. Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN
        
        Args:
            x: Input tensor (batch_size, sequence_length, 1)
            
        Returns:
            Output tensor (batch_size, sequence_length, num_classes)
        """
        batch_size, seq_length, _ = x.shape
        
        # Step 1: Prepare for 1D convolution
        # PyTorch Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, 1, seq_length)
        
        # Step 2: Project input
        x = self.input_projection(x)  # (batch, hidden_size, seq_length)
        
        # Step 3: Pass through TCN blocks
        # Each block captures patterns at different time scales
        residual = x
        
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
            # Residual connection every few blocks
            if tcn_block == self.tcn_blocks[len(self.tcn_blocks)//2]:
                x = x + residual
                residual = x
        
        # Step 4: Convert back to (batch, seq, features) format
        x = x.transpose(1, 2)  # (batch, seq_length, hidden_size)
        
        # Step 5: Apply layer normalization
        x = self.layer_norm(x)
        
        # Step 6: Classify each time step
        output = self.classifier(x)  # (batch, seq_length, num_classes)
        
        return output
    
    def get_model_info(self) -> dict:
        """Returns information about the model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        receptive_field = 1 + sum(2**i * (self.kernel_size - 1) for i in range(self.num_levels))
        
        return {
            'model_type': 'TCN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_size': self.hidden_size,
            'num_levels': self.num_levels,
            'kernel_size': self.kernel_size,
            'receptive_field': receptive_field,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate
        }

class TCNBlock(nn.Module):
    """
    Individual TCN block with residual connections
    This is a building block for the TCN model.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int, dropout: float):
        super(TCNBlock, self).__init__()
        
        # Padding to maintain sequence length
        padding = (kernel_size - 1) * dilation
        
        # Dilated convolution
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        
        # Another convolution for depth
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        
        # Crop padding (causal convolution)
        self.crop = padding
        
        # Normalization and activation
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input for residual connection
        residual = x
        
        # First convolution block
        x = self.conv1(x)
        if self.crop > 0:
            x = x[:, :, :-self.crop]  # Remove padding
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second convolution block
        x = self.conv2(x)
        if self.crop > 0:
            x = x[:, :, :-self.crop]  # Remove padding
        x = self.batch_norm2(x)
        
        # Residual connection
        if self.residual_conv:
            residual = self.residual_conv(residual)
        
        # Ensure same length for addition
        min_length = min(x.size(2), residual.size(2))
        x = x[:, :, :min_length]
        residual = residual[:, :, :min_length]
        
        x = x + residual
        x = self.activation(x)
        
        return x

def create_model(model_type: str, config) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: 'bilstm' or 'tcn'
        config: Configuration object
        
    Returns:
        Initialized model
    """
    if model_type.lower() == 'bilstm':
        model = BiLSTMSwaraClassifier(config)
        print("🧠 Created Bi-LSTM model")
    elif model_type.lower() == 'tcn':
        model = TCNSwaraClassifier(config)
        print("🔥 Created TCN model")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move to appropriate device
    model = model.to(config.DEVICE)
    
    # Print model information
    info = model.get_model_info()
    print(f"📊 Model Info: {info['model_type']}")
    print(f"   Parameters: {info['total_parameters']:,}")
    print(f"   Classes: {info['num_classes']}")
    print(f"   Device: {config.DEVICE}")
    
    return model

# Example usage and testing
if __name__ == "__main__":
    print("🧪 TESTING MODEL ARCHITECTURES")
    print("=" * 40)
    
    from saraga_setup import SaragaConfig
    
    config = SaragaConfig()
    
    # Test both models
    models_to_test = ['bilstm', 'tcn']
    
    for model_type in models_to_test:
        print(f"\n🔬 Testing {model_type.upper()} model...")
        
        # Create model
        model = create_model(model_type, config)
        
        # Create dummy input (batch_size=2, seq_length=100, features=1)
        dummy_input = torch.randn(2, 100, 1).to(config.DEVICE)
        
        # Test forward pass
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ Forward pass successful!")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Expected output shape: (2, 100, {model.num_classes})")
        
        # Test model summary
        info = model.get_model_info()
        print(f"📋 Model summary:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    print("\n✅ All model tests passed!")
    print("🚀 Models are ready for training!")
