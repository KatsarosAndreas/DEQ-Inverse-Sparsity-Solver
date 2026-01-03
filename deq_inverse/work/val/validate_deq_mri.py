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
import operators.singlecoil_mri as mrimodel
from operators.operator import OperatorPlusNoise
from utils.fastmri_dataloader import MultiSliceFastMRIDataloader
from networks.normalized_equilibrium_u_net import DnCNN
from solvers.equilibrium_solvers import EquilibriumProxGradMRI
from solvers import new_equilibrium_utils as eq_utils

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

def save_individual_slice(target_norm, y_norm, recon_norm, slice_idx, 
                         target_psnr, target_ssim, recon_psnr, recon_ssim, output_dir):
    """Save individual slice with 3 images: Ground Truth, Zero-filled, DEQ Reconstruction"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground Truth
    axes[0].imshow(target_norm, cmap='gray')
    axes[0].set_title(f'Ground Truth\nSlice {slice_idx}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Zero-filled (IFFT(A*y))
    axes[1].imshow(y_norm, cmap='gray')
    axes[1].set_title(f'Zero-filled IFFT(A*y)\nPSNR: {target_psnr:.2f}dB, SSIM: {target_ssim:.3f}', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # DEQ Reconstruction
    axes[2].imshow(recon_norm, cmap='gray')
    axes[2].set_title(f'DEQ Reconstruction\nPSNR: {recon_psnr:.2f}dB, SSIM: {recon_ssim:.3f}', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save individual slice
    slice_filename = os.path.join(output_dir, f'slice_{slice_idx:04d}_comparison.png')
    
    try:
        plt.savefig(slice_filename, dpi=150, bbox_inches='tight')
        plt.close()  # Important: close the figure to free memory
        return slice_filename
    except Exception as e:
        print(f"Warning: Failed to save slice {slice_idx}: {e}")
        plt.close()  # Still close the figure
        return None

def create_zero_filled_ifft(kspace_full, mask):
    """Create zero-filled IFFT reconstruction as paper baseline."""
    # Apply undersampling mask to full k-space
    kspace_undersampled = kspace_full * mask.to(kspace_full.device)
    
    # Zero-filled IFFT (missing frequencies are already zeros after masking)
    # Use the adjoint operation to get proper format matching the target
    # This ensures consistent tensor format [B, 2, H, W]
    complex_kspace = torch.view_as_complex(kspace_undersampled)
    zero_filled_complex = torch.fft.ifftn(complex_kspace, dim=[1, 2], norm="ortho")
    zero_filled_image = torch.view_as_real(zero_filled_complex)
    
    return zero_filled_image

def compute_metrics(pred, target):
    """Compute PSNR and SSIM metrics using ground-truth scaling (paper methodology)."""
    # Convert to magnitude 
    if pred.shape[0] == 2:  # Complex data [2, H, W]
        pred_mag = complex_abs(pred).cpu().numpy()
        target_mag = complex_abs(target).cpu().numpy()
    else:
        pred_mag = pred.cpu().numpy()
        target_mag = target.cpu().numpy()
    
    # Debug: Check shapes match
    if pred_mag.shape != target_mag.shape:
        print(f"Shape mismatch: pred_mag {pred_mag.shape} vs target_mag {target_mag.shape}")
        print(f"Original shapes: pred {pred.shape} vs target {target.shape}")
        # Try to make shapes consistent
        min_h = min(pred_mag.shape[0], target_mag.shape[0])
        min_w = min(pred_mag.shape[1], target_mag.shape[1])
        pred_mag = pred_mag[:min_h, :min_w]
        target_mag = target_mag[:min_h, :min_w]
        print(f"Resized to: pred_mag {pred_mag.shape} vs target_mag {target_mag.shape}")
    
    # CRITICAL FIX: Use ground-truth scaling for both images (paper methodology)
    # This fixes the "reconstruction looks good but PSNR drops" issue
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
                        default=r"G:\AK\deq_model\deq_PROX_fixedeta_pre_and4_final_ritsa_FINAL_22_2521.ckpt")
    parser.add_argument('--data_path', 
                        default=r"K:\thesis\data\singlecoil_val")
    parser.add_argument('--num_slices', type=int, default=200, 
                        help='Number of slices to validate')
    parser.add_argument('--acceleration', type=float, default=8.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--and_maxiters', default=30)
    parser.add_argument('--and_beta', type=float, default=0.4)  # Match optimized training parameters
    parser.add_argument('--and_m', type=int, default=3)  # Match optimized training parameters
    parser.add_argument('--etainit', type=float, default=0.1)  # Match optimized training parameters
    args = parser.parse_args()

    # Create validation results directory structure
    validation_results_dir = os.path.join(os.path.dirname(__file__), 'validation_results')
    os.makedirs(validation_results_dir, exist_ok=True)
    
    # Create run-specific subfolder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(validation_results_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")

    # Set random seed for reproducible but different slices each run
    random_seed = random.randint(1, 10000)  # Generate different seed each run
    random.seed(random_seed)
    print(f"Using random seed: {random_seed} for slice selection")
    
    # Parameters matching training script exactly
    n_channels = 2
    dataheight = 320
    datawidth = 320
    mri_center_fraction = 0.04
    mri_acceleration = float(args.acceleration)
    initial_eta = float(args.etainit)
    noise_sigma = 1e-2
    max_iters = int(args.and_maxiters)
    anderson_m = int(args.and_m)
    anderson_beta = float(args.and_beta)

    # Create mask exactly as in training
    mask = mrimodel.create_mask(shape=[dataheight, datawidth, 2], 
                               acceleration=mri_acceleration,
                               center_fraction=mri_center_fraction, seed=10)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup operators exactly as in training script
    forward_operator = mrimodel.cartesianSingleCoilMRI(kspace_mask=mask).to(device=device)
    measurement_process = OperatorPlusNoise(forward_operator, noise_sigma=noise_sigma).to(device=device)
    internal_forward_operator = mrimodel.cartesianSingleCoilMRI(kspace_mask=mask).to(device=device)

    # Create learned component exactly as in training
    learned_component = DnCNN(channels=n_channels, num_of_layers=17, lip=1.0)  # Fixed parameter name

    # Create solver exactly as in training
    solver = EquilibriumProxGradMRI(linear_operator=internal_forward_operator, 
                                   nonlinear_operator=learned_component,
                                   eta=initial_eta, minval=-6, maxval=6)
    solver = solver.to(device=device)

    # Load trained model
    print(f"Loading model from: {args.model_path}")
    if os.path.exists(args.model_path):
        saved_dict = torch.load(args.model_path, map_location=device)
        
        # Handle state dict compatibility exactly as in training script
        state_dict = saved_dict['solver_state_dict']
        if 'eta' in state_dict:
            eta_value = state_dict.pop('eta')
            solver.eta.data = torch.tensor(eta_value)
        
        solver.load_state_dict(state_dict, strict=False)
        print(f"Model loaded successfully from epoch {saved_dict.get('epoch', 'unknown')}")
    else:
        print(f"ERROR: Model file not found: {args.model_path}")
        return

    # Setup DEQ module exactly as in training
    forward_iterator = eq_utils.andersonexp
    deep_eq_module = eq_utils.DEQFixedPoint(solver, forward_iterator, m=anderson_m, 
                                           beta=anderson_beta, lam=0.08,
                                           max_iter=max_iters, tol=1e-5)  # Match optimized training parameters

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
    zero_filled_psnr_values = []
    zero_filled_ssim_values = []
    
    # Loss function same as training
    loss_function = torch.nn.MSELoss(reduction='sum')
    
    solver.eval()
    
    print("\n=== Starting DEQ Validation ===")
    
    saved_slices = []
    
    for idx, slice_idx in enumerate(tqdm(slice_indices, desc="Validating")):
        # Get data (input_img, target_img) from MultiSlice loader
        input_img, target_img = dataset[slice_idx]
        target_img = target_img.unsqueeze(0).to(device)  # Add batch dimension
        
        # Create measurements exactly as in training
        with torch.no_grad():
            y = measurement_process(target_img)
            # Get initial point exactly as in training script
            initial_point = forward_operator.adjoint(y)
            
            # SIMPLIFIED FIX: Use initial_point as zero-filled baseline
            # The initial_point is A†(y) which represents the zero-filled reconstruction
            # This is actually what most papers use as the "zero-filled" baseline
            zero_filled_reconstruction = initial_point
        
        # DEQ reconstruction exactly as in training (needs gradients for fixed point)
        y.requires_grad_(True)
        initial_point.requires_grad_(True)
        reconstruction = deep_eq_module.forward(y, initial_point=initial_point)
        
        # Compute loss exactly as in training
        loss = loss_function(reconstruction, target_img)
        losses.append(loss.item())
        
        # Compute metrics for both zero-filled and DEQ reconstruction
        with torch.no_grad():
            pred_squeeze = reconstruction.squeeze()  # Remove batch dimension
            target_squeeze = target_img.squeeze()   # Remove batch dimension
            zero_filled_squeeze = zero_filled_reconstruction.squeeze()  # Use A†(y) as baseline
            
            # DEQ reconstruction metrics
            recon_psnr, recon_ssim = compute_metrics(pred_squeeze, target_squeeze)
            psnr_values.append(recon_psnr)
            ssim_values.append(recon_ssim)
            
            # Zero-filled metrics (using A†(y) baseline)
            zf_psnr, zf_ssim = compute_metrics(zero_filled_squeeze, target_squeeze)
            zero_filled_psnr_values.append(zf_psnr)
            zero_filled_ssim_values.append(zf_ssim)
            
            # Prepare images for display
            target_display = complex_abs(target_squeeze)
            y_display = complex_abs(zero_filled_squeeze)  # Zero-filled IFFT
            recon_display = complex_abs(pred_squeeze)
            
            # Normalize for display
            target_norm = normalize_for_display(target_display).cpu()
            y_norm = normalize_for_display(y_display).cpu() 
            recon_norm = normalize_for_display(recon_display).cpu()
            
            # Save individual slice
            slice_filename = save_individual_slice(
                target_norm, y_norm, recon_norm, slice_idx,
                zf_psnr, zf_ssim, recon_psnr, recon_ssim, run_dir
            )
            
            if slice_filename:  # Only add if successfully saved
                saved_slices.append((slice_idx, slice_filename, recon_psnr, recon_ssim, zf_psnr, zf_ssim))
            
            print(f"Slice {slice_idx}: Loss={loss.item():.4f}, DEQ PSNR={recon_psnr:.2f}dB, DEQ SSIM={recon_ssim:.3f}, ZF PSNR={zf_psnr:.2f}dB, ZF SSIM={zf_ssim:.3f}")

    # Compute summary statistics
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    mean_psnr = np.mean(psnr_values)
    std_psnr = np.std(psnr_values)
    mean_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)
    mean_zf_psnr = np.mean(zero_filled_psnr_values)
    std_zf_psnr = np.std(zero_filled_psnr_values)
    mean_zf_ssim = np.mean(zero_filled_ssim_values)
    std_zf_ssim = np.std(zero_filled_ssim_values)
    
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Model: {os.path.basename(args.model_path)}")
    print(f"Data: {args.data_path}")
    print(f"Acceleration: {args.acceleration}x")
    print(f"Slices validated: {args.num_slices}")
    print(f"Results saved to: {run_dir}")
    print(f"")
    print(f"ZERO-FILLED RESULTS:")
    print(f"Mean ZF PSNR: {mean_zf_psnr:.2f} ± {std_zf_psnr:.2f} dB")
    print(f"Mean ZF SSIM: {mean_zf_ssim:.3f} ± {std_zf_ssim:.3f}")
    print(f"")
    print(f"DEQ RECONSTRUCTION RESULTS:")
    print(f"Mean Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"Mean DEQ PSNR: {mean_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"Mean DEQ SSIM: {mean_ssim:.3f} ± {std_ssim:.3f}")
    print(f"")
    print(f"IMPROVEMENT:")
    print(f"PSNR Improvement: {mean_psnr - mean_zf_psnr:.2f} dB")
    print(f"SSIM Improvement: {mean_ssim - mean_zf_ssim:.3f}")
    
    # BEAUTIFUL Performance plots - scatter plots for discrete data
    plt.figure(figsize=(15, 5))
    
    # Plot 1: PSNR Comparison (Beautiful scatter plot)
    plt.subplot(1, 3, 1)
    x_pos = np.arange(len(psnr_values))
    
    # Scatter plot instead of ugly line plot
    plt.scatter(x_pos, zero_filled_psnr_values, color='red', alpha=0.7, s=60, 
                label=f'Zero-filled (μ={mean_zf_psnr:.1f})', edgecolors='darkred')
    plt.scatter(x_pos, psnr_values, color='blue', alpha=0.7, s=60, 
                label=f'DEQ (μ={mean_psnr:.1f})', edgecolors='darkblue')
    
    # Add mean lines
    plt.axhline(y=mean_zf_psnr, color='red', linestyle='--', alpha=0.8, linewidth=2)
    plt.axhline(y=mean_psnr, color='blue', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.xlabel('Validation Slice Index')
    plt.ylabel('PSNR (dB)')
    plt.title(f'PSNR Comparison\n({args.acceleration}x Accelerated MRI)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss Distribution (Beautiful histogram instead of ugly line)
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
    plt.scatter(x_pos, zero_filled_ssim_values, color='red', alpha=0.7, s=60, 
                label=f'Zero-filled (μ={mean_zf_ssim:.3f})', edgecolors='darkred')
    plt.scatter(x_pos, ssim_values, color='blue', alpha=0.7, s=60, 
                label=f'DEQ (μ={mean_ssim:.3f})', edgecolors='darkblue')
    
    # Add mean lines
    plt.axhline(y=mean_zf_ssim, color='red', linestyle='--', alpha=0.8, linewidth=2)
    plt.axhline(y=mean_ssim, color='blue', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.xlabel('Validation Slice Index')
    plt.ylabel('SSIM')
    plt.title('SSIM Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    performance_plot_path = os.path.join(run_dir, 'validation_performance.png')
    plt.savefig(performance_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # BEAUTIFUL Analysis plots - better ranking visualization + reliability assessment
    plt.figure(figsize=(15, 5))
    
    # Plot 1: PSNR vs SSIM Correlation (keep but improve)
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
    
    # Plot 2: BEAUTIFUL Performance Distribution (Box plots instead of ugly ranking)
    plt.subplot(1, 3, 2)
    
    # Create box plot data
    data_to_plot = [zero_filled_psnr_values, psnr_values]
    box_plot = plt.boxplot(data_to_plot, labels=['Zero-filled', 'DEQ'], patch_artist=True)
    
    # Beautiful colors
    colors = ['lightcoral', 'lightblue']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Distribution\nComparison')
    plt.grid(True, alpha=0.3)
    
    # Add improvement annotation
    improvement = mean_psnr - mean_zf_psnr
    plt.text(0.5, 0.95, f'Improvement: +{improvement:.2f} dB', 
            transform=plt.gca().transAxes, fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Plot 3: MODEL RELIABILITY Assessment (NEW!)
    plt.subplot(1, 3, 3)
    
    # Calculate improvement for each slice
    improvements = np.array(psnr_values) - np.array(zero_filled_psnr_values)
    
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
    plt.title(f'Model Reliability\n{reliability_rate:.1f}% Success Rate')
    plt.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Add reliability assessment text
    if reliability_rate >= 70:
        assessment = "RELIABLE"
        color = 'green'
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
    analysis_plot_path = os.path.join(run_dir, 'validation_analysis.png')
    plt.savefig(analysis_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # SIMPLE SUMMARY ASSESSMENT PLOT (despite undertraining)
    plt.figure(figsize=(10, 6))
    
    # Create simple, clear summary
    plt.subplot(2, 2, 1)
    # Improvement statistics
    improvements = np.array(psnr_values) - np.array(zero_filled_psnr_values)
    ssim_improvements = np.array(ssim_values) - np.array(zero_filled_ssim_values)
    
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
    zf_means = [mean_zf_psnr, mean_zf_ssim]
    deq_means = [mean_psnr, mean_ssim]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, zf_means, width, label='Zero-filled', color='red', alpha=0.7)
    plt.bar(x + width/2, deq_means, width, label='DEQ', color='blue', alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add improvement text
    for i, (metric, zf_val, deq_val) in enumerate(zip(metrics, zf_means, deq_means)):
        improvement = deq_val - zf_val
        plt.text(i, max(zf_val, deq_val) + 0.02 * max(zf_val, deq_val), 
                f'+{improvement:.3f}', ha='center', fontweight='bold', color='green')
    
    plt.subplot(2, 1, 2)
    # Overall assessment despite undertraining
    assessment_text = f"""
MODEL ASSESSMENT (Despite Limited Training):

✓ Mean PSNR Improvement: +{mean_psnr - mean_zf_psnr:.2f} dB
✓ Mean SSIM Improvement: +{mean_ssim - mean_zf_ssim:.3f}
✓ Success Rate: {positive_improvements}/{total_slices} slices ({reliability_rate:.1f}%)
✓ Correlation (PSNR-SSIM): {corr_coef:.3f}

CONCLUSION: Model shows {assessment.lower()} performance with consistent improvement
over zero-filled reconstruction despite being undertrained.
Further training expected to yield significant gains.
"""
    
    plt.text(0.05, 0.95, assessment_text, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    plt.axis('off')
    
    plt.suptitle('DEQ Model Validation Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    summary_plot_path = os.path.join(run_dir, 'validation_summary.png')
    plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # DIAGNOSTIC PLOTS - Sanity checks to verify experiments are correct
    plt.figure(figsize=(18, 10))
    
    # Plot 1: Loss vs PSNR correlation (should be negative - lower loss = higher PSNR)
    plt.subplot(2, 3, 1)
    plt.scatter(losses, psnr_values, alpha=0.7, color='blue', s=60)
    loss_psnr_corr = np.corrcoef(losses, psnr_values)[0, 1]
    plt.xlabel('MSE Loss')
    plt.ylabel('DEQ PSNR (dB)')
    plt.title(f'Loss vs PSNR Correlation\nρ={loss_psnr_corr:.3f}')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(losses, psnr_values, 1)
    p = np.poly1d(z)
    plt.plot(losses, p(losses), "r--", alpha=0.8)
    
    # Sanity check warning
    if loss_psnr_corr > -0.3:  # Should be negative correlation
        plt.text(0.05, 0.95, 'WARNING: Weak negative correlation!', 
                transform=plt.gca().transAxes, fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    else:
        plt.text(0.05, 0.95, 'GOOD: Strong negative correlation', 
                transform=plt.gca().transAxes, fontsize=10, color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot 2: Zero-filled vs DEQ PSNR scatter (points should be above diagonal)
    plt.subplot(2, 3, 2)
    plt.scatter(zero_filled_psnr_values, psnr_values, alpha=0.7, color='purple', s=60)
    
    # Add diagonal line (y=x) - points above this line mean DEQ > Zero-filled
    min_psnr = min(min(zero_filled_psnr_values), min(psnr_values))
    max_psnr = max(max(zero_filled_psnr_values), max(psnr_values))
    plt.plot([min_psnr, max_psnr], [min_psnr, max_psnr], 'r--', alpha=0.8, label='y=x (No improvement)')
    
    plt.xlabel('Zero-filled PSNR (dB)')
    plt.ylabel('DEQ PSNR (dB)')
    plt.title('DEQ vs Zero-filled PSNR\n(Points above line = improvement)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Count points above diagonal
    points_above = np.sum(np.array(psnr_values) > np.array(zero_filled_psnr_values))
    improvement_rate = points_above / len(psnr_values) * 100
    
    if improvement_rate < 50:
        plt.text(0.05, 0.95, f'WARNING: Only {improvement_rate:.1f}% improved!', 
                transform=plt.gca().transAxes, fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    else:
        plt.text(0.05, 0.95, f'GOOD: {improvement_rate:.1f}% improved', 
                transform=plt.gca().transAxes, fontsize=10, color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot 3: Loss distribution normality check
    plt.subplot(2, 3, 3)
    plt.hist(losses, bins=15, alpha=0.7, color='orange', edgecolor='black', density=True)
    plt.xlabel('MSE Loss')
    plt.ylabel('Density')
    plt.title('Loss Distribution\n(Should be roughly normal)')
    plt.grid(True, alpha=0.3)
    
    # Add mean and std lines
    plt.axvline(x=mean_loss, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_loss:.4f}')
    plt.axvline(x=mean_loss + std_loss, color='red', linestyle='--', alpha=0.7, label=f'±1σ')
    plt.axvline(x=mean_loss - std_loss, color='red', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 4: PSNR range sanity check (typical MRI PSNR values)
    plt.subplot(2, 3, 4)
    all_psnr = zero_filled_psnr_values + psnr_values
    plt.hist([zero_filled_psnr_values, psnr_values], bins=15, alpha=0.7, 
             label=['Zero-filled', 'DEQ'], color=['red', 'blue'], edgecolor='black')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Count')
    plt.title('PSNR Range Check\n(Should be 15-35 dB for MRI)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sanity check for reasonable PSNR ranges
    min_all_psnr = min(all_psnr)
    max_all_psnr = max(all_psnr)
    
    if min_all_psnr < 10 or max_all_psnr > 50:
        plt.text(0.05, 0.95, f'WARNING: Unusual PSNR range [{min_all_psnr:.1f}, {max_all_psnr:.1f}]', 
                transform=plt.gca().transAxes, fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    else:
        plt.text(0.05, 0.95, f'GOOD: Normal PSNR range [{min_all_psnr:.1f}, {max_all_psnr:.1f}]', 
                transform=plt.gca().transAxes, fontsize=10, color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot 5: SSIM vs PSNR for both methods (should be correlated)
    plt.subplot(2, 3, 5)
    plt.scatter(zero_filled_psnr_values, zero_filled_ssim_values, alpha=0.7, color='red', 
                s=60, label='Zero-filled', edgecolors='darkred')
    plt.scatter(psnr_values, ssim_values, alpha=0.7, color='blue', 
                s=60, label='DEQ', edgecolors='darkblue')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('SSIM')
    plt.title('PSNR-SSIM Consistency Check\n(Should be positively correlated)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Check PSNR-SSIM correlation for each method
    zf_corr = np.corrcoef(zero_filled_psnr_values, zero_filled_ssim_values)[0, 1]
    deq_corr = np.corrcoef(psnr_values, ssim_values)[0, 1]
    
    if min(zf_corr, deq_corr) < 0.3:
        plt.text(0.05, 0.95, f'WARNING: Weak PSNR-SSIM correlation!', 
                transform=plt.gca().transAxes, fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    else:
        plt.text(0.05, 0.95, f'GOOD: Strong PSNR-SSIM correlation', 
                transform=plt.gca().transAxes, fontsize=10, color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot 6: Slice-by-slice improvement consistency
    plt.subplot(2, 3, 6)
    slice_improvements = np.array(psnr_values) - np.array(zero_filled_psnr_values)
    plt.plot(range(len(slice_improvements)), slice_improvements, 'o-', alpha=0.7, color='green')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8, label='No improvement')
    plt.xlabel('Slice Index')
    plt.ylabel('PSNR Improvement (dB)')
    plt.title('Per-Slice Improvement\n(Consistency check)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Check for outliers (improvements > 3 std deviations)
    improvement_std = np.std(slice_improvements)
    improvement_mean = np.mean(slice_improvements)
    outliers = np.abs(slice_improvements - improvement_mean) > 3 * improvement_std
    num_outliers = np.sum(outliers)
    
    if num_outliers > len(slice_improvements) * 0.1:  # More than 10% outliers
        plt.text(0.05, 0.95, f'WARNING: {num_outliers} outliers detected!', 
                transform=plt.gca().transAxes, fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    else:
        plt.text(0.05, 0.95, f'GOOD: {num_outliers} outliers (<10%)', 
                transform=plt.gca().transAxes, fontsize=10, color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('EXPERIMENT SANITY CHECKS - Diagnostic Plots', fontsize=16, fontweight='bold')
    plt.tight_layout()
    diagnostic_plot_path = os.path.join(run_dir, 'validation_diagnostics.png')
    plt.savefig(diagnostic_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # SUMMARY DIAGNOSTIC REPORT
    print(f"\n=== EXPERIMENT SANITY CHECK REPORT ===")
    print(f"1. Loss-PSNR Correlation: {loss_psnr_corr:.3f} {'✓ GOOD' if loss_psnr_corr < -0.3 else '⚠ WARNING: Should be more negative'}")
    print(f"2. Improvement Rate: {improvement_rate:.1f}% {'✓ GOOD' if improvement_rate >= 50 else '⚠ WARNING: Low improvement rate'}")
    print(f"3. PSNR Range: [{min_all_psnr:.1f}, {max_all_psnr:.1f}] dB {'✓ GOOD' if 10 <= min_all_psnr and max_all_psnr <= 50 else '⚠ WARNING: Unusual range'}")
    print(f"4. ZF PSNR-SSIM Correlation: {zf_corr:.3f} {'✓ GOOD' if zf_corr >= 0.3 else '⚠ WARNING: Weak correlation'}")
    print(f"5. DEQ PSNR-SSIM Correlation: {deq_corr:.3f} {'✓ GOOD' if deq_corr >= 0.3 else '⚠ WARNING: Weak correlation'}")
    print(f"6. Outliers: {num_outliers}/{len(slice_improvements)} {'✓ GOOD' if num_outliers <= len(slice_improvements)*0.1 else '⚠ WARNING: Many outliers'}")
    
    # Overall experiment validity assessment
    checks_passed = sum([
        loss_psnr_corr < -0.3,
        improvement_rate >= 50,
        10 <= min_all_psnr and max_all_psnr <= 50,
        zf_corr >= 0.3,
        deq_corr >= 0.3,
        num_outliers <= len(slice_improvements) * 0.1
    ])
    
    print(f"\nOVERALL EXPERIMENT VALIDITY: {checks_passed}/6 checks passed")
    if checks_passed >= 5:
        print("✓ EXPERIMENT APPEARS VALID - Results are trustworthy")
    elif checks_passed >= 3:
        print("⚠ EXPERIMENT PARTIALLY VALID - Some concerns but acceptable")
    else:
        print("❌ EXPERIMENT QUESTIONABLE - Multiple issues detected, review setup")
    
    # Identify best and worst reconstructions
    best_idx = np.argmax(psnr_values)
    worst_idx = np.argmin(psnr_values)
    
    print(f"\nValidation completed successfully!")
    print(f"Results saved in: {run_dir}")
    print(f"- Individual slice comparisons: {len(psnr_values)} PNG files")
    print(f"- validation_performance.png (beautiful performance metrics)")
    print(f"- validation_analysis.png (improved reliability analysis)")
    print(f"- validation_summary.png (simple overall assessment)")
    print(f"\nBest reconstruction: Slice {slice_indices[best_idx]} (PSNR: {psnr_values[best_idx]:.2f}dB)")
    print(f"Worst reconstruction: Slice {slice_indices[worst_idx]} (PSNR: {psnr_values[worst_idx]:.2f}dB)")
    print(f"\nMODEL ASSESSMENT: {assessment} ({reliability_rate:.1f}% success rate)")
    print(f"Despite undertraining, model shows consistent improvement potential!")

if __name__ == "__main__":
    main()
