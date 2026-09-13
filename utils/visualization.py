import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import io
from PIL import Image

def plot_saliency_map(saliency_map, title):
    """
    Generates a saliency map plot and returns it as a W&B image.
    Handles both 1D and 2D saliency maps.
    """
    # Handle tensor conversion safely
    if hasattr(saliency_map, 'cpu'):
        saliency_np = saliency_map.cpu().detach().numpy()
    else:
        saliency_np = saliency_map
    
    # Handle 0-dimensional arrays
    if saliency_np.ndim == 0:
        saliency_np = saliency_np.reshape(1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(saliency_np.shape) == 1:
        # 1D saliency map - plot as line
        ax.plot(saliency_np)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Attribution')
        ax.grid(True, alpha=0.3)
    elif len(saliency_np.shape) == 2:
        # 2D saliency map - plot as heatmap
        im = ax.imshow(saliency_np, cmap='hot', interpolation='nearest', aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Features')
    else:
        # Handle higher dimensional cases by flattening
        saliency_flat = saliency_np.flatten()
        ax.plot(saliency_flat)
        ax.set_xlabel('Flattened Index')
        ax.set_ylabel('Attribution')
        ax.grid(True, alpha=0.3)
    
    ax.set_title(title)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    
    return wandb.Image(image)

def plot_activation_histogram(activations, title):
    """
    Generates a histogram of activations and returns it as a W&B image.
    """
    fig, ax = plt.subplots()
    ax.hist(activations.cpu().numpy().flatten(), bins=50)
    ax.set_title(title)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)

    return wandb.Image(image)


def plot_patches_per_patient_histogram(patch_counts_per_patient, title="Patches per Patient Distribution"):
    """
    Generates a histogram showing distribution of patch counts per patient.

    Args:
        patch_counts_per_patient: pandas Series with patient_id as index and patch counts as values
        title: Title for the plot

    Returns:
        wandb.Image: Histogram plot as W&B image
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram of patch counts
    counts = patch_counts_per_patient.values
    ax.hist(counts, bins=20, edgecolor='black', alpha=0.7)

    # Add statistics to the plot
    mean_patches = np.mean(counts)
    median_patches = np.median(counts)

    ax.axvline(mean_patches, color='red', linestyle='--', label=f'Mean: {mean_patches:.1f}')
    ax.axvline(median_patches, color='blue', linestyle='--', label=f'Median: {median_patches:.1f}')

    ax.set_xlabel('Number of Patches')
    ax.set_ylabel('Number of Patients')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text box with summary statistics
    stats_text = f'Total Patients: {len(patch_counts_per_patient)}\nMin: {np.min(counts)}\nMax: {np.max(counts)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)

    return wandb.Image(image)
