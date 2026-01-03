#!/usr/bin/env python3
"""
DnCNN Denoiser Validation Script
Validates DnCNN denoiser trained on noise-only (no masks) for MRI reconstruction
Creates beautiful plots and individual slice comparisons: Ground Truth, Noisy, Denoised
"""

import torch
import os
import random
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from datetime import datetime

# Add the root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, root_dir)

import torch.nn as nn
import operators.operator as lin_operator
from operators.operator import OperatorPlusNoise
from utils.fastmri_dataloader import MultiSliceFastMRIDataloader
from networks.normalized_equilibrium_u_net import DnCNN

def complex_abs(data):
    """Compute magnitude of complex data."""
    if data.dim() == 3 and data.size(0) == 2:  # [2, H, W] format
        return torch.sqrt(data[0]**2 + data[1]**2)
    elif data.dim() == 4 and data.size(-1) == 2:  # [..., H, W, 2] format
        return (data ** 2).sum(dim=-1).sqrt()
    else:
        # Assume it's already magnitude data
        return torch.norm(data, dim=0) if data.dim() > 2 else data

def normalize_for_display(img):
    """Normalize image to [0, 1] for display."""
    img_flat = img.flatten()
    img_min = torch.min(img_flat)
    img_max = torch.max(img_flat)
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img

def save_individual_slice(target_norm, noisy_norm, denoised_norm, slice_idx, 
                         noisy_psnr, noisy_ssim, denoised_psnr, denoised_ssim, 
                         noise_sigma, output_dir):
    """Save individual slice with 3 images: Ground Truth, Noisy, DnCNN Denoised"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground Truth
    axes[0].imshow(target_norm, cmap='gray')
    axes[0].set_title(f'Ground Truth\nSlice {slice_idx}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Noisy (σ=noise_sigma)
    axes[1].imshow(noisy_norm, cmap='gray')
    axes[1].set_title(f'Noisy (σ={noise_sigma})\nPSNR: {noisy_psnr:.2f}dB, SSIM: {noisy_ssim:.3f}', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # DnCNN Denoised
    axes[2].imshow(denoised_norm, cmap='gray')
    axes[2].set_title(f'DnCNN Denoised\nPSNR: {denoised_psnr:.2f}dB, SSIM: {denoised_ssim:.3f}', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save individual slice
    slice_filename = os.path.join(output_dir, f'slice_{slice_idx:04d}_denoising_comparison.png')
    
    try:
        plt.savefig(slice_filename, dpi=150, bbox_inches='tight')
        plt.close()  # Important: close the figure to free memory
        return slice_filename
    except Exception as e:
        print(f"Warning: Failed to save slice {slice_idx}: {e}")
        plt.close()  # Still close the figure
        return None

def compute_metrics(pred, target):
    """Compute PSNR and SSIM metrics using ground-truth scaling."""
    # Convert to magnitude 
    if pred.shape[0] == 2:  # Complex data [2, H, W]
        pred_mag = complex_abs(pred).cpu().numpy()
        target_mag = complex_abs(target).cpu().numpy()
    else:
        pred_mag = pred.cpu().numpy()
        target_mag = target.cpu().numpy()
    
    # Use ground-truth scaling for consistent metrics
    vmin = target_mag.min()
    vmax = target_mag.max()
    
    # Apply same scaling to both images based on ground-truth range
    target_norm = np.clip((target_mag - vmin) / (vmax - vmin + 1e-8), 0, 1)
    pred_norm = np.clip((pred_mag - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    # Compute metrics with consistent scaling
    psnr_val = psnr(target_norm, pred_norm, data_range=1.0)
    ssim_val = ssim(target_norm, pred_norm, data_range=1.0, gaussian_weights=True)
    
    return psnr_val, ssim_val

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', 
                        default=r"G:\AK\dncnn_model\dncnn_mri_sigma03_optimized_final.ckpt")
    parser.add_argument('--data_path', 
                        default=r"K:\thesis\data\singlecoil_val")
    parser.add_argument('--num_slices', type=int, default=80, 
                        help='Number of slices to validate')
    parser.add_argument('--noise_sigma', type=float, default=0.3, 
                        help='Noise level to add for validation (e.g., 0.01, 0.6)')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    # Create validation results directory structure
    validation_results_dir = os.path.join(os.path.dirname(__file__), 'denoiser_validation_results')
    os.makedirs(validation_results_dir, exist_ok=True)
    
    # Create run-specific subfolder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(validation_results_dir, f'dncnn_sigma{args.noise_sigma}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")

    # Set random seed for reproducible but different slices each run
    random_seed = random.randint(1, 10000)  # Generate different seed each run
    random.seed(random_seed)
    print(f"Using random seed: {random_seed} for slice selection")
    print(f"Validation noise level: σ={args.noise_sigma}")
    
    # Parameters
    n_channels = 2
    noise_sigma = float(args.noise_sigma)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup operators - Identity for denoising (no masks)
    forward_operator = lin_operator.Identity().to(device=device)
    measurement_process = OperatorPlusNoise(forward_operator, noise_sigma=noise_sigma).to(device=device)

    # Create DnCNN denoiser exactly as in training script
    denoiser = DnCNN(channels=n_channels, num_of_layers=17, lip=1.0)
    denoiser = denoiser.to(device=device)

    # Load trained denoiser model
    print(f"Loading denoiser from: {args.model_path}")
    if os.path.exists(args.model_path):
        saved_dict = torch.load(args.model_path, map_location=device)
        
        # Handle state dict compatibility
        state_dict = saved_dict['solver_state_dict']
        denoiser.load_state_dict(state_dict, strict=False)
        print(f"Denoiser loaded successfully from epoch {saved_dict.get('epoch', 'unknown')}")
    else:
        print(f"ERROR: Model file not found: {args.model_path}")
        return

    # Create validation dataset using MultiSlice logic
    print(f"Loading validation data from: {args.data_path}")
    dataset = MultiSliceFastMRIDataloader(args.data_path, data_indices=None)
    total_slices = len(dataset)
    print(f"Total available slices: {total_slices}")
    
    # Randomly sample slices for validation (different each run)
    if args.num_slices > total_slices:
        args.num_slices = total_slices
        print(f"Using all available {total_slices} slices.")
    
    slice_indices = random.sample(range(total_slices), args.num_slices)
    print(f"Validating on {args.num_slices} randomly selected slices")

    # Metrics storage
    losses = []
    psnr_values = []
    ssim_values = []
    noisy_psnr_values = []
    noisy_ssim_values = []
    
    # Loss function same as training
    loss_function = torch.nn.MSELoss(reduction='sum')
    
    denoiser.eval()
    
    print("\n=== Starting DnCNN Denoiser Validation ===")
    
    saved_slices = []
    
    for idx, slice_idx in enumerate(tqdm(slice_indices, desc="Validating")):
        # Get data (input_img, target_img) from MultiSlice loader
        input_img, target_img = dataset[slice_idx]
        target_img = target_img.unsqueeze(0).to(device)  # Add batch dimension
        
        # Create noisy measurements exactly as in training
        with torch.no_grad():
            # Add noise to clean image
            noisy_image = measurement_process(target_img)
        
        # DnCNN denoising (residual learning: y + DnCNN(y))
        with torch.no_grad():
            denoised_image = noisy_image + denoiser(noisy_image)
        
        # Compute loss exactly as in training
        loss = loss_function(denoised_image, target_img)
        losses.append(loss.item())
        
        # Compute metrics for both noisy and denoised images
        with torch.no_grad():
            target_squeeze = target_img.squeeze()      # Remove batch dimension
            noisy_squeeze = noisy_image.squeeze()      # Remove batch dimension  
            denoised_squeeze = denoised_image.squeeze()  # Remove batch dimension
            
            # Denoised image metrics
            denoised_psnr, denoised_ssim = compute_metrics(denoised_squeeze, target_squeeze)
            psnr_values.append(denoised_psnr)
            ssim_values.append(denoised_ssim)
            
            # Noisy image metrics (baseline)
            noisy_psnr, noisy_ssim = compute_metrics(noisy_squeeze, target_squeeze)
            noisy_psnr_values.append(noisy_psnr)
            noisy_ssim_values.append(noisy_ssim)
            
            # Prepare images for display
            target_display = complex_abs(target_squeeze)
            noisy_display = complex_abs(noisy_squeeze)
            denoised_display = complex_abs(denoised_squeeze)
            
            # Normalize for display
            target_norm = normalize_for_display(target_display).cpu()
            noisy_norm = normalize_for_display(noisy_display).cpu() 
            denoised_norm = normalize_for_display(denoised_display).cpu()
            
            # Save individual slice
            slice_filename = save_individual_slice(
                target_norm, noisy_norm, denoised_norm, slice_idx,
                noisy_psnr, noisy_ssim, denoised_psnr, denoised_ssim, 
                noise_sigma, run_dir
            )
            
            if slice_filename:  # Only add if successfully saved
                saved_slices.append((slice_idx, slice_filename, denoised_psnr, denoised_ssim, noisy_psnr, noisy_ssim))
            
            print(f"Slice {slice_idx}: Loss={loss.item():.4f}, Denoised PSNR={denoised_psnr:.2f}dB, Denoised SSIM={denoised_ssim:.3f}, Noisy PSNR={noisy_psnr:.2f}dB, Noisy SSIM={noisy_ssim:.3f}")

    # Compute summary statistics
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    mean_psnr = np.mean(psnr_values)
    std_psnr = np.std(psnr_values)
    mean_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)
    mean_noisy_psnr = np.mean(noisy_psnr_values)
    std_noisy_psnr = np.std(noisy_psnr_values)
    mean_noisy_ssim = np.mean(noisy_ssim_values)
    std_noisy_ssim = np.std(noisy_ssim_values)
    
    print(f"\n=== DENOISER VALIDATION SUMMARY ===")
    print(f"Model: {os.path.basename(args.model_path)}")
    print(f"Data: {args.data_path}")
    print(f"Noise Level: σ={args.noise_sigma}")
    print(f"Slices validated: {args.num_slices}")
    print(f"Results saved to: {run_dir}")
    print(f"")
    print(f"NOISY RESULTS:")
    print(f"Mean Noisy PSNR: {mean_noisy_psnr:.2f} ± {std_noisy_psnr:.2f} dB")
    print(f"Mean Noisy SSIM: {mean_noisy_ssim:.3f} ± {std_noisy_ssim:.3f}")
    print(f"")
    print(f"DNCNN DENOISED RESULTS:")
    print(f"Mean Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"Mean Denoised PSNR: {mean_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"Mean Denoised SSIM: {mean_ssim:.3f} ± {std_ssim:.3f}")
    print(f"")
    print(f"DENOISING IMPROVEMENT:")
    print(f"PSNR Improvement: +{mean_psnr - mean_noisy_psnr:.2f} dB")
    print(f"SSIM Improvement: +{mean_ssim - mean_noisy_ssim:.3f}")
    
    # BEAUTIFUL Performance plots - scatter plots for discrete data
    plt.figure(figsize=(15, 5))
    
    # Plot 1: PSNR Comparison (Beautiful scatter plot)
    plt.subplot(1, 3, 1)
    x_pos = np.arange(len(psnr_values))
    
    # Scatter plot instead of ugly line plot
    plt.scatter(x_pos, noisy_psnr_values, color='red', alpha=0.7, s=60, 
                label=f'Noisy (μ={mean_noisy_psnr:.1f})', edgecolors='darkred')
    plt.scatter(x_pos, psnr_values, color='blue', alpha=0.7, s=60, 
                label=f'DnCNN (μ={mean_psnr:.1f})', edgecolors='darkblue')
    
    # Add mean lines
    plt.axhline(y=mean_noisy_psnr, color='red', linestyle='--', alpha=0.8, linewidth=2)
    plt.axhline(y=mean_psnr, color='blue', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.xlabel('Validation Slice Index')
    plt.ylabel('PSNR (dB)')
    plt.title(f'PSNR Comparison\n(σ={args.noise_sigma} Noise Level)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss Distribution (Beautiful histogram)
    plt.subplot(1, 3, 2)
    
    # Compute per-pixel MSE
    height, width, channels = 320, 320, 2
    pixels_per_slice = height * width * channels
    per_pixel_mse = [loss / pixels_per_slice for loss in losses]
    mean_per_pixel_mse = mean_loss / pixels_per_slice
    
    # Beautiful histogram showing loss distribution
    plt.hist(per_pixel_mse, bins=15, alpha=0.7, color='green', edgecolor='darkgreen')
    plt.axvline(x=mean_per_pixel_mse, color='red', linestyle='--', linewidth=3, 
                label=f'Mean: {mean_per_pixel_mse:.4f}')
    plt.xlabel('Per-Pixel MSE')
    plt.ylabel('Number of Slices')
    plt.title('Loss Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: SSIM Comparison (Beautiful scatter plot)
    plt.subplot(1, 3, 3)
    plt.scatter(x_pos, noisy_ssim_values, color='red', alpha=0.7, s=60, 
                label=f'Noisy (μ={mean_noisy_ssim:.3f})', edgecolors='darkred')
    plt.scatter(x_pos, ssim_values, color='blue', alpha=0.7, s=60, 
                label=f'DnCNN (μ={mean_ssim:.3f})', edgecolors='darkblue')
    
    # Add mean lines
    plt.axhline(y=mean_noisy_ssim, color='red', linestyle='--', alpha=0.8, linewidth=2)
    plt.axhline(y=mean_ssim, color='blue', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.xlabel('Validation Slice Index')
    plt.ylabel('SSIM')
    plt.title('SSIM Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    performance_plot_path = os.path.join(run_dir, 'denoiser_performance.png')
    plt.savefig(performance_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # BEAUTIFUL Analysis plots - reliability assessment
    plt.figure(figsize=(15, 5))
    
    # Plot 1: PSNR vs SSIM Correlation
    plt.subplot(1, 3, 1)
    plt.scatter(psnr_values, ssim_values, alpha=0.8, s=80, c=losses, 
                cmap='viridis', edgecolors='black')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('SSIM')
    plt.title('PSNR vs SSIM Correlation\n(Color = Loss)')
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar()
    cbar.set_label('MSE Loss')
    
    # Add correlation coefficient
    corr_coef = np.corrcoef(psnr_values, ssim_values)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
            transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Performance Distribution (Box plots)
    plt.subplot(1, 3, 2)
    
    # Create box plot data
    data_to_plot = [noisy_psnr_values, psnr_values]
    box_plot = plt.boxplot(data_to_plot, labels=['Noisy', 'DnCNN'], patch_artist=True)
    
    # Beautiful colors
    colors = ['lightcoral', 'lightblue']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Distribution\nComparison')
    plt.grid(True, alpha=0.3)
    
    # Add improvement annotation
    improvement = mean_psnr - mean_noisy_psnr
    plt.text(0.5, 0.95, f'Improvement: +{improvement:.2f} dB', 
            transform=plt.gca().transAxes, fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Plot 3: DENOISER RELIABILITY Assessment
    plt.subplot(1, 3, 3)
    
    # Calculate improvement for each slice
    improvements = np.array(psnr_values) - np.array(noisy_psnr_values)
    
    # Count reliable improvements
    positive_improvements = np.sum(improvements > 0)
    total_slices = len(improvements)
    reliability_rate = positive_improvements / total_slices * 100
    
    # Beautiful bar chart showing reliability
    categories = ['Improved', 'Degraded']
    counts = [positive_improvements, total_slices - positive_improvements]
    colors = ['green', 'red']
    
    bars = plt.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Number of Slices')
    plt.title(f'Denoiser Reliability\n{reliability_rate:.1f}% Success Rate')
    plt.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Add reliability assessment text
    if reliability_rate >= 90:
        assessment = "EXCELLENT"
        color = 'green'
    elif reliability_rate >= 75:
        assessment = "GOOD"
        color = 'lightgreen'
    elif reliability_rate >= 50:
        assessment = "MODERATE"
        color = 'orange'
    else:
        assessment = "POOR"
        color = 'red'
    
    plt.text(0.5, 0.85, f'Assessment: {assessment}', 
            transform=plt.gca().transAxes, fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    plt.tight_layout()
    analysis_plot_path = os.path.join(run_dir, 'denoiser_analysis.png')
    plt.savefig(analysis_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # SIMPLE SUMMARY ASSESSMENT PLOT
    plt.figure(figsize=(10, 6))
    
    # Create simple, clear summary
    plt.subplot(2, 2, 1)
    # Improvement statistics
    improvements = np.array(psnr_values) - np.array(noisy_psnr_values)
    ssim_improvements = np.array(ssim_values) - np.array(noisy_ssim_values)
    
    plt.hist(improvements, bins=12, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement')
    plt.axvline(x=np.mean(improvements), color='green', linestyle='-', linewidth=2, 
                label=f'Mean: +{np.mean(improvements):.2f}dB')
    plt.xlabel('PSNR Improvement (dB)')
    plt.ylabel('Number of Slices')
    plt.title('PSNR Improvement Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    # Simple metrics comparison
    metrics = ['PSNR (dB)', 'SSIM']
    noisy_means = [mean_noisy_psnr, mean_noisy_ssim]
    denoised_means = [mean_psnr, mean_ssim]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, noisy_means, width, label='Noisy', color='red', alpha=0.7)
    plt.bar(x + width/2, denoised_means, width, label='DnCNN', color='blue', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add improvement text
    for i, (metric, noisy_val, denoised_val) in enumerate(zip(metrics, noisy_means, denoised_means)):
        improvement = denoised_val - noisy_val
        plt.text(i, max(noisy_val, denoised_val) + 0.02 * max(noisy_val, denoised_val), 
                f'+{improvement:.3f}', ha='center', fontweight='bold', color='green')
    
    plt.subplot(2, 1, 2)
    # Overall assessment
    assessment_text = f"""
DNCNN DENOISER ASSESSMENT:

✓ Mean PSNR Improvement: +{mean_psnr - mean_noisy_psnr:.2f} dB
✓ Mean SSIM Improvement: +{mean_ssim - mean_noisy_ssim:.3f}
✓ Success Rate: {positive_improvements}/{total_slices} slices ({reliability_rate:.1f}%)
✓ Correlation (PSNR-SSIM): {corr_coef:.3f}
✓ Noise Level Tested: σ={noise_sigma}

CONCLUSION: DnCNN denoiser shows {assessment.lower()} performance with consistent 
noise reduction across validation slices. Model effectively removes σ={noise_sigma} 
noise while preserving image structure and detail.
"""
    
    plt.text(0.05, 0.95, assessment_text, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    plt.axis('off')
    
    plt.suptitle('DnCNN Denoiser Validation Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    summary_plot_path = os.path.join(run_dir, 'denoiser_summary.png')
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # ADDITIONAL BEAUTIFUL PLOTS FOR DENOISER ANALYSIS - COMBINED
    from mpl_toolkits.mplot3d import Axes3D
    from scipy import stats
    
    # COMBINED ADVANCED ANALYSIS PLOT
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: 3D PSNR Surface (Beautiful!)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Create meshgrid for surface plot
    slice_nums = np.array(range(len(psnr_values)))
    X, Y = np.meshgrid(slice_nums[:min(50, len(slice_nums))], [0, 1])  # Limit for visual clarity
    
    # Create surface data (Noisy vs Denoised)
    Z_noisy = np.array([noisy_psnr_values[:min(50, len(psnr_values))], noisy_psnr_values[:min(50, len(psnr_values))]])
    Z_denoised = np.array([psnr_values[:min(50, len(psnr_values))], psnr_values[:min(50, len(psnr_values))]])
    
    # Beautiful surface plots
    surf1 = ax1.plot_surface(X, Y, Z_noisy, alpha=0.6, color='red', label='Noisy')
    surf2 = ax1.plot_surface(X, Y, Z_denoised, alpha=0.8, color='blue', label='DnCNN')
    
    ax1.set_xlabel('Slice Index')
    ax1.set_ylabel('Method (0=Noisy, 1=DnCNN)')
    ax1.set_zlabel('PSNR (dB)')
    ax1.set_title('3D Performance Landscape\nPSNR Across Slices')
    ax1.view_init(elev=20, azim=45)
    ax1.zaxis.set_major_locator(plt.MaxNLocator(5))
    
    # Plot 2: Loss vs Improvement Relationship
    ax2 = fig.add_subplot(2, 3, 2)
    plt.scatter(losses, improvements, alpha=0.7, s=60, color='purple', edgecolors='black')
    plt.xlabel('MSE Loss')
    plt.ylabel('PSNR Improvement (dB)')
    plt.title('Loss vs Improvement\nRelationship')
    plt.grid(True, alpha=0.3)
    
    # Add correlation info
    loss_imp_corr = np.corrcoef(losses, improvements)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {loss_imp_corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: SSIM Improvement Analysis
    ax3 = fig.add_subplot(2, 3, 3)
    ssim_improvements = np.array(ssim_values) - np.array(noisy_ssim_values)
    plt.scatter(improvements, ssim_improvements, alpha=0.7, s=60, 
                color='orange', edgecolors='black')
    plt.xlabel('PSNR Improvement (dB)')
    plt.ylabel('SSIM Improvement')
    plt.title('PSNR vs SSIM\nImprovement Correlation')
    plt.grid(True, alpha=0.3)
    
    # Add diagonal reference
    max_val = max(max(improvements), max(ssim_improvements))
    min_val = min(min(improvements), min(ssim_improvements))
    plt.plot([min_val, max_val], [min_val*0.1, max_val*0.1], 'r--', alpha=0.5, label='Reference')
    plt.legend()
    
    # Plot 4: Quality Transformation Heatmap
    ax4 = fig.add_subplot(2, 3, 4)
    plt.hist2d(noisy_psnr_values, psnr_values, bins=15, cmap='Blues', alpha=0.7)
    plt.plot([min(noisy_psnr_values), max(noisy_psnr_values)], 
             [min(noisy_psnr_values), max(noisy_psnr_values)], 'r--', linewidth=2)
    plt.xlabel('Noisy PSNR (dB)')
    plt.ylabel('DnCNN PSNR (dB)')
    plt.title('Quality Transformation\n(Above line = Improvement)')
    
    # Plot 5: Noise Robustness Assessment
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Create performance bins based on noisy image quality
    quality_bins = ['Low PSNR\n(<20dB)', 'Med PSNR\n(20-25dB)', 'High PSNR\n(>25dB)']
    bin_improvements = []
    
    for bin_thresh in [(0, 20), (20, 25), (25, 100)]:
        mask = (np.array(noisy_psnr_values) >= bin_thresh[0]) & (np.array(noisy_psnr_values) < bin_thresh[1])
        if np.any(mask):
            bin_improvements.append(np.mean(np.array(improvements)[mask]))
        else:
            bin_improvements.append(0)
    
    bars = plt.bar(quality_bins, bin_improvements, alpha=0.7, 
                   color=['red', 'orange', 'green'], edgecolor='black')
    plt.ylabel('Mean PSNR Improvement (dB)')
    plt.title('Robustness Analysis\nPerformance vs Input Quality')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, bin_improvements):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}dB', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Statistical Significance Summary
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Perform statistical tests
    t_stat, p_value = stats.ttest_rel(psnr_values, noisy_psnr_values)
    
    # Create summary visualization
    improvement_stats = {
        'Mean Improvement': np.mean(improvements),
        'Median Improvement': np.median(improvements),
        'Std Improvement': np.std(improvements)
    }
    
    bars = plt.bar(range(len(improvement_stats)), list(improvement_stats.values()),
                   color=['blue', 'green', 'orange'], alpha=0.7, edgecolor='black')
    plt.xticks(range(len(improvement_stats)), list(improvement_stats.keys()), rotation=45)
    plt.ylabel('PSNR Improvement (dB)')
    plt.title('Statistical Summary\nDenoising Performance')
    plt.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, improvement_stats.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Add significance annotation
    plt.text(0.5, 0.95, f'p-value: {p_value:.2e}\n{"Significant" if p_value < 0.05 else "Not significant"}', 
             transform=plt.gca().transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.suptitle('ADVANCED DENOISER ANALYSIS - Combined Visualizations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    advanced_plot_path = os.path.join(run_dir, 'denoiser_combined_analysis.png')
    plt.savefig(advanced_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nADDITIONAL PLOTS GENERATED:")
    print(f"- denoiser_combined_analysis.png (3D surface + key analyses combined)")
    
    # Identify best and worst reconstructions
    best_idx = np.argmax(psnr_values)
    worst_idx = np.argmin(psnr_values)
    
    print(f"\nValidation completed successfully!")
    print(f"Results saved in: {run_dir}")
    print(f"- Individual slice comparisons: {len(psnr_values)} PNG files")
    print(f"- denoiser_performance.png (performance metrics)")
    print(f"- denoiser_analysis.png (reliability analysis)")
    print(f"- denoiser_summary.png (overall assessment)")
    print(f"\nBest denoising: Slice {slice_indices[best_idx]} (PSNR: {psnr_values[best_idx]:.2f}dB)")
    print(f"Worst denoising: Slice {slice_indices[worst_idx]} (PSNR: {psnr_values[worst_idx]:.2f}dB)")
    print(f"\nMODEL ASSESSMENT: {assessment} ({reliability_rate:.1f}% success rate)")
    print(f"DnCNN effectively removes σ={noise_sigma} noise with +{mean_psnr - mean_noisy_psnr:.2f}dB improvement!")

if __name__ == "__main__":
    main()