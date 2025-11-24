"""
Model Visualization Tools
Inspired by ScrimBrain - Visualize CNN filters and activations for debugging
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Optional, List, Dict, Tuple
from half_sword_ai.core.model import HalfSwordPolicyNetwork

logger = logging.getLogger(__name__)

class ModelVisualizer:
    """
    Visualize CNN filters and activations
    Similar to ScrimBrain's visualization tools for debugging model behavior
    """
    
    def __init__(self, model: HalfSwordPolicyNetwork):
        self.model = model
        self.model.eval()  # Set to eval mode for visualization
    
    def visualize_conv_filters(self, layer_name: str = 'conv1', output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize convolutional filters for a given layer
        
        Args:
            layer_name: Name of conv layer ('conv1', 'conv2', 'conv3', 'conv4')
            output_path: Path to save visualization (optional)
        """
        # Get conv layer
        conv_layer = getattr(self.model, layer_name, None)
        if conv_layer is None:
            logger.error(f"Layer {layer_name} not found")
            return None
        
        # Get filters (weights)
        filters = conv_layer.weight.data.cpu().numpy()
        
        # Get filter dimensions
        n_filters, n_channels, h, w = filters.shape
        
        # Create visualization grid
        n_cols = min(8, n_filters)
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_filters):
            row = i // n_cols
            col = i % n_cols
            
            # Get filter (average across input channels for visualization)
            filter_img = filters[i]
            if n_channels > 1:
                filter_img = np.mean(filter_img, axis=0)  # Average across channels
            else:
                filter_img = filter_img[0]
            
            # Normalize for visualization
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
            
            # Plot
            axes[row, col].imshow(filter_img, cmap='gray')
            axes[row, col].set_title(f'Filter {i}')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_filters, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved filter visualization to {output_path}")
        
        return fig
    
    def visualize_activations(self, frame: np.ndarray, layer_names: List[str] = None, 
                            output_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Visualize activations for given layers with a sample frame
        
        Args:
            frame: Input frame (H, W) or (T, H, W) for frame stack
            layer_names: List of layer names to visualize (default: all conv layers)
            output_path: Directory to save visualizations (optional)
        """
        if layer_names is None:
            layer_names = ['conv1', 'conv2', 'conv3', 'conv4']
        
        # Prepare frame input
        if len(frame.shape) == 2:
            # Single frame, create frame stack
            frame_stack = np.stack([frame] * self.model.frame_stack_size, axis=0)
        elif len(frame.shape) == 3:
            frame_stack = frame
        else:
            logger.error(f"Invalid frame shape: {frame.shape}")
            return {}
        
        # Convert to tensor
        frame_tensor = torch.FloatTensor(frame_stack).unsqueeze(0)  # Add batch dimension
        frame_tensor = frame_tensor.to(next(self.model.parameters()).device)
        
        # Register hooks to capture activations
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach().cpu().numpy()
            return hook
        
        hooks = []
        for layer_name in layer_names:
            layer = getattr(self.model, layer_name, None)
            if layer is not None:
                hooks.append(layer.register_forward_hook(get_activation(layer_name)))
        
        # Forward pass
        with torch.no_grad():
            state_features = torch.zeros(1, 7).to(frame_tensor.device)  # Dummy state features
            self.model(frame_tensor, state_features)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Visualize activations
        visualizations = {}
        for layer_name, activation in activations.items():
            # Get activation shape: (batch, channels, H, W)
            if len(activation.shape) == 4:
                act = activation[0]  # Remove batch dimension
                n_channels = act.shape[0]
                
                # Select a few channels to visualize
                n_vis = min(16, n_channels)
                selected_channels = np.linspace(0, n_channels - 1, n_vis, dtype=int)
                
                # Create grid
                n_cols = min(4, n_vis)
                n_rows = (n_vis + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
                if n_rows == 1:
                    axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
                
                for idx, channel_idx in enumerate(selected_channels):
                    row = idx // n_cols
                    col = idx % n_cols
                    
                    channel_act = act[channel_idx]
                    # Normalize for visualization
                    channel_act = (channel_act - channel_act.min()) / (channel_act.max() - channel_act.min() + 1e-8)
                    
                    if n_rows == 1 and n_cols == 1:
                        axes.imshow(channel_act, cmap='hot')
                        axes.set_title(f'{layer_name} Ch {channel_idx}')
                        axes.axis('off')
                    else:
                        axes[row, col].imshow(channel_act, cmap='hot')
                        axes[row, col].set_title(f'Ch {channel_idx}')
                        axes[row, col].axis('off')
                
                # Hide unused subplots
                for idx in range(len(selected_channels), n_rows * n_cols):
                    row = idx // n_cols
                    col = idx % n_cols
                    if n_rows == 1:
                        if isinstance(axes, list):
                            continue
                        axes[col].axis('off')
                    else:
                        axes[row, col].axis('off')
                
                plt.tight_layout()
                visualizations[layer_name] = fig
                
                if output_path:
                    layer_path = Path(output_path) / f"{layer_name}_activations.png"
                    plt.savefig(layer_path, dpi=150, bbox_inches='tight')
                    logger.info(f"Saved {layer_name} activations to {layer_path}")
                    plt.close(fig)
        
        return visualizations
    
    def visualize_feature_map(self, frame: np.ndarray, output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize complete feature extraction pipeline
        
        Args:
            frame: Input frame
            output_path: Path to save visualization
        """
        # Prepare frame
        if len(frame.shape) == 2:
            frame_stack = np.stack([frame] * self.model.frame_stack_size, axis=0)
        elif len(frame.shape) == 3:
            frame_stack = frame
        else:
            return None
        
        frame_tensor = torch.FloatTensor(frame_stack).unsqueeze(0)
        frame_tensor = frame_tensor.to(next(self.model.parameters()).device)
        
        # Forward pass through conv layers
        with torch.no_grad():
            x = frame_tensor
            x1 = torch.relu(self.model.conv1(x))
            x2 = torch.relu(self.model.conv2(x1))
            x3 = torch.relu(self.model.conv3(x2))
            x4 = torch.relu(self.model.conv4(x3))
            
            # Get feature maps (first channel of each layer)
            feature_maps = [
                x1[0, 0].cpu().numpy(),
                x2[0, 0].cpu().numpy(),
                x3[0, 0].cpu().numpy(),
                x4[0, 0].cpu().numpy()
            ]
        
        # Visualize
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        # Original frame
        if len(frame.shape) == 3:
            original = np.mean(frame, axis=0)  # Average frame stack
        else:
            original = frame
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Frame')
        axes[0].axis('off')
        
        # Feature maps
        for i, feat_map in enumerate(feature_maps):
            feat_map_norm = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
            axes[i+1].imshow(feat_map_norm, cmap='hot')
            axes[i+1].set_title(f'Conv{i+1}')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved feature map visualization to {output_path}")
        
        return fig
    
    def get_layer_info(self) -> Dict[str, Dict]:
        """Get information about all layers in the model"""
        info = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_info = {
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters()),
                }
                
                if isinstance(module, nn.Conv2d):
                    layer_info.update({
                        'in_channels': module.in_channels,
                        'out_channels': module.out_channels,
                        'kernel_size': module.kernel_size,
                        'stride': module.stride,
                        'padding': module.padding
                    })
                elif isinstance(module, nn.Linear):
                    layer_info.update({
                        'in_features': module.in_features,
                        'out_features': module.out_features
                    })
                
                info[name] = layer_info
        
        return info

def visualize_model_checkpoint(checkpoint_path: str, frame: np.ndarray, output_dir: str):
    """
    Load model checkpoint and visualize it
    
    Args:
        checkpoint_path: Path to model checkpoint
        frame: Sample frame for activation visualization
        output_dir: Directory to save visualizations
    """
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = HalfSwordPolicyNetwork()
    
    # Load weights (handle different checkpoint formats)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    # Create visualizer
    visualizer = ModelVisualizer(model)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Visualize filters
    logger.info("Visualizing convolutional filters...")
    for layer_name in ['conv1', 'conv2', 'conv3', 'conv4']:
        visualizer.visualize_conv_filters(
            layer_name=layer_name,
            output_path=str(output_path / f"{layer_name}_filters.png")
        )
    
    # Visualize activations
    logger.info("Visualizing activations...")
    visualizer.visualize_activations(
        frame=frame,
        output_path=str(output_path)
    )
    
    # Visualize feature maps
    logger.info("Visualizing feature maps...")
    visualizer.visualize_feature_map(
        frame=frame,
        output_path=str(output_path / "feature_maps.png")
    )
    
    # Print layer info
    logger.info("Model layer information:")
    layer_info = visualizer.get_layer_info()
    for name, info in layer_info.items():
        logger.info(f"  {name}: {info}")
    
    logger.info(f"✅ Visualizations saved to {output_path}")

