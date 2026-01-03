#!/usr/bin/env python3
"""
BEAUTIFUL MSE vs ⊥-loss Model Comparison Script
Compares two DEQ models trained with different loss functions on SAME EXACT slices
Creates paper-quality plots showing performance differences with statistical significance
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
from scipy import stats

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
        return torch.norm(data, dim=0) if data.dim() > 2 else data

def normalize_for_display(img):
    """Normalize image to [0, 1] for display."""
    img_flat = img.flatten()
    img_min = torch.min(img_flat)
    img_max = torch.max(img_flat)
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img

def compute_metrics(pred, target):
    """Compute PSNR and SSIM metrics using ground-truth scaling."""
    # Convert to magnitude 
    if pred.shape[0] == 2:  # Complex data [2, H, W]
        pred_mag = complex_abs(pred).cpu().numpy()
        target_mag = complex_abs(target).cpu().numpy()
    else:
        pred_mag = pred.cpu().numpy()
        target_mag = target.cpu().numpy()
    
    # Ground-truth scaling for consistent metrics
    vmin = target_mag.min()
    vmax = target_mag.max()
    
    target_norm = np.clip((target_mag - vmin) / (vmax - vmin + 1e-8), 0, 1)
    pred_norm = np.clip((pred_mag - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    psnr_val = psnr(target_norm, pred_norm, data_range=1.0)
    ssim_val = ssim(target_norm, pred_norm, data_range=1.0, gaussian_weights=True)
    
    return psnr_val, ssim_val

def load_model(model_path, device, mask):
    """Load a DEQ model from checkpoint."""
    # Create the model architecture
    learned_component = DnCNN(2, num_of_layers=17, lip=1.0)
    forward_operator = mrimodel.cartesianSingleCoilMRI(kspace_mask=mask).to(device=device)
    solver = EquilibriumProxGradMRI(linear_operator=forward_operator, 
                                   nonlinear_operator=learned_component,
                                   eta=0.25, minval=-6, maxval=6)
    solver = solver.to(device=device)
    
    # Load checkpoint
    if os.path.exists(model_path):
        saved_dict = torch.load(model_path, map_location=device)
        state_dict = saved_dict['solver_state_dict']
        
        # Handle eta parameter if it exists
        if 'eta' in state_dict:
            eta_value = state_dict.pop('eta')
            solver.eta.data = torch.tensor(eta_value)
        
        solver.load_state_dict(state_dict, strict=False)
        print(f"Loaded model: {os.path.basename(model_path)}")
        
        return solver
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

def save_comparison_slice(target_norm, zf_norm, mse_norm, perp_norm, slice_idx, 
                         zf_psnr, zf_ssim, mse_psnr, mse_ssim, perp_psnr, perp_ssim, output_dir):
    """Save comparison slice with 4 images: Ground Truth, Zero-filled, MSE, ⊥-loss"""
    
    plt.figure(figsize=(16, 4))
    
    # Ground Truth
    plt.subplot(1, 4, 1)
    plt.imshow(target_norm, cmap='gray')
    plt.title('Ground Truth', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Zero-filled
    plt.subplot(1, 4, 2)
    plt.imshow(zf_norm, cmap='gray')
    plt.title(f'Zero-filled\nPSNR: {zf_psnr:.2f}dB\nSSIM: {zf_ssim:.3f}', fontsize=11)
    plt.axis('off')
    
    # MSE Model
    plt.subplot(1, 4, 3)
    plt.imshow(mse_norm, cmap='gray')
    improvement_mse = mse_psnr - zf_psnr
    plt.title(f'MSE Loss\nPSNR: {mse_psnr:.2f}dB (+{improvement_mse:.2f})\nSSIM: {mse_ssim:.3f}', 
              fontsize=11, color='blue')
    plt.axis('off')
    
    # ⊥-loss Model
    plt.subplot(1, 4, 4)
    plt.imshow(perp_norm, cmap='gray')
    improvement_perp = perp_psnr - zf_psnr
    advantage = perp_psnr - mse_psnr
    color = 'green' if advantage > 0 else 'red'
    plt.title(f'⊥-loss\nPSNR: {perp_psnr:.2f}dB (+{improvement_perp:.2f})\nSSIM: {perp_ssim:.3f}', 
              fontsize=11, color=color)
    plt.axis('off')
    
    plt.suptitle(f'Slice {slice_idx} Comparison - ⊥-loss vs MSE: +{advantage:.2f}dB', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    slice_filename = os.path.join(output_dir, f'comparison_slice_{slice_idx:03d}.png')
    plt.savefig(slice_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return slice_filename

def create_comparison_plots(results, run_dir, args):
    """Create beautiful comparison plots showing MSE vs ⊥-loss performance."""
    
    # Plot 1: Side-by-side performance comparison
    plt.figure(figsize=(15, 5))
    
    # PSNR Comparison
    plt.subplot(1, 3, 1)
    methods = ['Zero-filled', 'MSE Loss', '⊥-loss']
    psnr_means = [np.mean(results['zf_psnr']), np.mean(results['mse_psnr']), np.mean(results['perp_psnr'])]
    psnr_stds = [np.std(results['zf_psnr']), np.std(results['mse_psnr']), np.std(results['perp_psnr'])]
    
    colors = ['red', 'blue', 'green']
    bars = plt.bar(methods, psnr_means, yerr=psnr_stds, capsize=5, 
                   color=colors, alpha=0.7, edgecolor='black')
    
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add improvement annotations
    mse_improvement = psnr_means[1] - psnr_means[0]
    perp_improvement = psnr_means[2] - psnr_means[0]
    
    plt.text(1, psnr_means[1] + psnr_stds[1] + 0.2, f'+{mse_improvement:.2f}', 
             ha='center', fontweight='bold', color='blue')
    plt.text(2, psnr_means[2] + psnr_stds[2] + 0.2, f'+{perp_improvement:.2f}', 
             ha='center', fontweight='bold', color='green')
    
    # SSIM Comparison
    plt.subplot(1, 3, 2)
    ssim_means = [np.mean(results['zf_ssim']), np.mean(results['mse_ssim']), np.mean(results['perp_ssim'])]
    ssim_stds = [np.std(results['zf_ssim']), np.std(results['mse_ssim']), np.std(results['perp_ssim'])]
    
    bars = plt.bar(methods, ssim_means, yerr=ssim_stds, capsize=5, 
                   color=colors, alpha=0.7, edgecolor='black')
    
    plt.ylabel('SSIM')
    plt.title('SSIM Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add improvement annotations
    mse_improvement = ssim_means[1] - ssim_means[0]
    perp_improvement = ssim_means[2] - ssim_means[0]
    
    plt.text(1, ssim_means[1] + ssim_stds[1] + 0.01, f'+{mse_improvement:.3f}', 
             ha='center', fontweight='bold', color='blue')
    plt.text(2, ssim_means[2] + ssim_stds[2] + 0.01, f'+{perp_improvement:.3f}', 
             ha='center', fontweight='bold', color='green')
    
    # Direct Comparison (⊥-loss advantage)
    plt.subplot(1, 3, 3)
    advantages_psnr = [p - m for p, m in zip(results['perp_psnr'], results['mse_psnr'])]
    
    plt.hist(advantages_psnr, bins=15, alpha=0.7, color='green', edgecolor='black', 
             label=f'PSNR (μ={np.mean(advantages_psnr):.2f})')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No advantage')
    plt.axvline(x=np.mean(advantages_psnr), color='green', linestyle='-', linewidth=2, 
                label=f'Mean: +{np.mean(advantages_psnr):.2f}dB')
    
    plt.xlabel('⊥-loss PSNR Advantage (dB)')
    plt.ylabel('Number of Slices')
    plt.title('⊥-loss vs MSE Advantage Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    wins = sum(1 for a in advantages_psnr if a > 0)
    plt.text(0.05, 0.95, f'Win Rate: {wins}/{len(advantages_psnr)} ({100*wins/len(advantages_psnr):.1f}%)', 
             transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle(f'MSE vs ⊥-loss Comparison ({args.acceleration}x Acceleration)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comparison_plot_path = os.path.join(run_dir, 'performance_comparison.png')
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Scatter plots showing correlation and differences
    plt.figure(figsize=(15, 5))
    
    # PSNR correlation
    plt.subplot(1, 3, 1)
    plt.scatter(results['mse_psnr'], results['perp_psnr'], alpha=0.7, s=60, 
                edgecolors='black', c='blue')
    
    # Add diagonal line (perfect correlation)
    min_val = min(min(results['mse_psnr']), min(results['perp_psnr']))
    max_val = max(max(results['mse_psnr']), max(results['perp_psnr']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    plt.xlabel('MSE Loss PSNR (dB)')
    plt.ylabel('⊥-loss PSNR (dB)')
    plt.title('PSNR Correlation')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(results['mse_psnr'], results['perp_psnr'])[0, 1]
    plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # SSIM correlation
    plt.subplot(1, 3, 2)
    plt.scatter(results['mse_ssim'], results['perp_ssim'], alpha=0.7, s=60, 
                edgecolors='black', c='green')
    
    min_val = min(min(results['mse_ssim']), min(results['perp_ssim']))
    max_val = max(max(results['mse_ssim']), max(results['perp_ssim']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    plt.xlabel('MSE Loss SSIM')
    plt.ylabel('⊥-loss SSIM')
    plt.title('SSIM Correlation')
    plt.grid(True, alpha=0.3)
    
    corr = np.corrcoef(results['mse_ssim'], results['perp_ssim'])[0, 1]
    plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Box plot comparison
    plt.subplot(1, 3, 3)
    data_to_plot = [results['mse_psnr'], results['perp_psnr']]
    box_plot = plt.boxplot(data_to_plot, labels=['MSE Loss', '⊥-loss'], patch_artist=True)
    
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Distribution Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add statistical test result
    _, pvalue = stats.ttest_rel(results['perp_psnr'], results['mse_psnr'])
    significance = "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else "ns"
    plt.text(0.5, 0.95, f'p = {pvalue:.4f} {significance}', 
             transform=plt.gca().transAxes, fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.suptitle(f'Statistical Analysis: MSE vs ⊥-loss', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    analysis_plot_path = os.path.join(run_dir, 'statistical_analysis.png')
    plt.savefig(analysis_plot_path, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mse_model', 
                        default=r"G:\AK\deq_model\deq_PROX_fixedeta_pre_and4_final_ritsa_FINAL_22.ckpt")
    parser.add_argument('--perp_model', 
                        default=r"G:\AK\deq_model\deq_PROX_fixedeta_pre_and_PREP_final_ritsa_FINALPREP_22.ckpt")
    parser.add_argument('--data_path', 
                        default=r"K:\thesis\data\singlecoil_val")
    parser.add_argument('--num_slices', type=int, default=125, 
                        help='Number of slices to compare')
    parser.add_argument('--acceleration', type=float, default=8.0)
    parser.add_argument('--seed', type=int, default=40, help='Random seed for slice selection')
    parser.add_argument('--and_maxiters', default=55)
    parser.add_argument('--and_beta', type=float, default=0.6)
    parser.add_argument('--and_m', type=int, default=6)
    args = parser.parse_args()

    # Set random seed for reproducible slice selection
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(__file__), 'comparison_results', f'mse_vs_perp_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")

    # Setup MRI parameters
    dataheight = 320
    datawidth = 320
    mri_center_fraction = 0.04
    mri_acceleration = args.acceleration
    noise_sigma = 1e-2

    # Create mask
    mask = mrimodel.create_mask(shape=[dataheight, datawidth, 2], 
                               acceleration=mri_acceleration,
                               center_fraction=mri_center_fraction, seed=10)

    # Setup operators
    forward_operator = mrimodel.cartesianSingleCoilMRI(kspace_mask=mask).to(device=device)
    measurement_process = OperatorPlusNoise(forward_operator, noise_sigma=noise_sigma).to(device=device)

    # Load both models
    print("\n=== Loading Models ===")
    mse_solver = load_model(args.mse_model, device, mask)
    perp_solver = load_model(args.perp_model, device, mask)

    # Setup DEQ modules for both models
    forward_iterator = eq_utils.andersonexp
    mse_deq_module = eq_utils.DEQFixedPoint(mse_solver, forward_iterator, 
                                           m=args.and_m, beta=args.and_beta, lam=0.08,
                                           max_iter=args.and_maxiters, tol=1e-4)
    perp_deq_module = eq_utils.DEQFixedPoint(perp_solver, forward_iterator,
                                            m=args.and_m, beta=args.and_beta, lam=0.08,
                                            max_iter=args.and_maxiters, tol=1e-5)

    # Load dataset
    dataset = MultiSliceFastMRIDataloader(args.data_path, data_indices=None)
    print(f"Dataset size: {len(dataset)}")

    # Select random slices (SAME for both models)
    total_slices = min(len(dataset), 1000)  # Limit for reasonable selection
    slice_indices = random.sample(range(total_slices), min(args.num_slices, total_slices))
    slice_indices.sort()  # Keep sorted for better organization
    
    print(f"Selected {len(slice_indices)} slices for comparison")
    print(f"First 10 slices: {slice_indices[:10]}...")

    # Set models to evaluation mode
    mse_solver.eval()
    perp_solver.eval()

    # Storage for results
    results = {
        'slice_indices': slice_indices,
        'zf_psnr': [], 'zf_ssim': [],
        'mse_psnr': [], 'mse_ssim': [],
        'perp_psnr': [], 'perp_ssim': []
    }

    saved_slices = []

    print("\n=== Starting Model Comparison ===")
    
    for idx, slice_idx in enumerate(tqdm(slice_indices, desc="Comparing MSE vs ⊥-loss")):
        # Get data
        input_img, target_img = dataset[slice_idx]
        target_img = target_img.unsqueeze(0).to(device)  # Add batch dimension
        
        # Create measurements (same for both models) - needs gradients for DEQ
        y = measurement_process(target_img)
        initial_point = forward_operator.adjoint(y)
        
        # Zero-filled baseline (no gradients needed)
        with torch.no_grad():
            zero_filled_reconstruction = initial_point.clone()
        
        # MSE model reconstruction (needs gradients for DEQ)
        y_mse = y.clone().requires_grad_(True)
        initial_mse = initial_point.clone().requires_grad_(True)
        mse_reconstruction = mse_deq_module.forward(y_mse, initial_point=initial_mse)
        
        # ⊥-loss model reconstruction (needs gradients for DEQ)
        y_perp = y.clone().requires_grad_(True)
        initial_perp = initial_point.clone().requires_grad_(True)
        perp_reconstruction = perp_deq_module.forward(y_perp, initial_point=initial_perp)
        
        # Compute metrics for all methods (no gradients needed)
        with torch.no_grad():
            target_squeeze = target_img.squeeze()
            zf_squeeze = zero_filled_reconstruction.squeeze()
            mse_squeeze = mse_reconstruction.detach().squeeze()  # Detach gradients
            perp_squeeze = perp_reconstruction.detach().squeeze()  # Detach gradients
            
            # Zero-filled metrics
            zf_psnr, zf_ssim = compute_metrics(zf_squeeze, target_squeeze)
            results['zf_psnr'].append(zf_psnr)
            results['zf_ssim'].append(zf_ssim)
            
            # MSE model metrics
            mse_psnr, mse_ssim = compute_metrics(mse_squeeze, target_squeeze)
            results['mse_psnr'].append(mse_psnr)
            results['mse_ssim'].append(mse_ssim)
            
            # ⊥-loss model metrics
            perp_psnr, perp_ssim = compute_metrics(perp_squeeze, target_squeeze)
            results['perp_psnr'].append(perp_psnr)
            results['perp_ssim'].append(perp_ssim)
            
            # Prepare images for display
            target_display = complex_abs(target_squeeze)
            zf_display = complex_abs(zf_squeeze)
            mse_display = complex_abs(mse_squeeze)
            perp_display = complex_abs(perp_squeeze)
            
            # Normalize for display
            target_norm = normalize_for_display(target_display).cpu()
            zf_norm = normalize_for_display(zf_display).cpu()
            mse_norm = normalize_for_display(mse_display).cpu()
            perp_norm = normalize_for_display(perp_display).cpu()
            
            # Save comparison slice
            slice_filename = save_comparison_slice(
                target_norm, zf_norm, mse_norm, perp_norm, slice_idx,
                zf_psnr, zf_ssim, mse_psnr, mse_ssim, perp_psnr, perp_ssim, run_dir
            )
            
            saved_slices.append((slice_idx, slice_filename, mse_psnr, mse_ssim, perp_psnr, perp_ssim))
            
            # Progress display
            advantage = perp_psnr - mse_psnr
            status = "+" if advantage > 0 else "-"
            print(f"Slice {slice_idx}: ZF={zf_psnr:.2f}dB, MSE={mse_psnr:.2f}dB, ⊥={perp_psnr:.2f}dB {status}+{advantage:.2f}dB")

    # Compute summary statistics
    zf_psnr_mean = np.mean(results['zf_psnr'])
    mse_psnr_mean = np.mean(results['mse_psnr'])
    perp_psnr_mean = np.mean(results['perp_psnr'])
    
    zf_ssim_mean = np.mean(results['zf_ssim'])
    mse_ssim_mean = np.mean(results['mse_ssim'])
    perp_ssim_mean = np.mean(results['perp_ssim'])
    
    # Calculate improvements
    mse_vs_zf_psnr = mse_psnr_mean - zf_psnr_mean
    perp_vs_zf_psnr = perp_psnr_mean - zf_psnr_mean
    perp_vs_mse_psnr = perp_psnr_mean - mse_psnr_mean
    
    mse_vs_zf_ssim = mse_ssim_mean - zf_ssim_mean
    perp_vs_zf_ssim = perp_ssim_mean - zf_ssim_mean
    perp_vs_mse_ssim = perp_ssim_mean - mse_ssim_mean
    
    # Statistical significance test
    _, pvalue_psnr = stats.ttest_rel(results['perp_psnr'], results['mse_psnr'])
    _, pvalue_ssim = stats.ttest_rel(results['perp_ssim'], results['mse_ssim'])
    
    # Count wins
    perp_wins_psnr = sum(1 for p, m in zip(results['perp_psnr'], results['mse_psnr']) if p > m)
    perp_wins_ssim = sum(1 for p, m in zip(results['perp_ssim'], results['mse_ssim']) if p > m)
    
    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"Models compared: MSE vs ⊥-loss")
    print(f"Slices compared: {len(slice_indices)}")
    print(f"Acceleration: {args.acceleration}x")
    print(f"")
    print(f"PSNR RESULTS:")
    print(f"Zero-filled:  {zf_psnr_mean:.2f} ± {np.std(results['zf_psnr']):.2f} dB")
    print(f"MSE Loss:     {mse_psnr_mean:.2f} ± {np.std(results['mse_psnr']):.2f} dB (+{mse_vs_zf_psnr:.2f})")
    print(f"⊥-loss:       {perp_psnr_mean:.2f} ± {np.std(results['perp_psnr']):.2f} dB (+{perp_vs_zf_psnr:.2f})")
    print(f"")
    print(f"SSIM RESULTS:")
    print(f"Zero-filled:  {zf_ssim_mean:.3f} ± {np.std(results['zf_ssim']):.3f}")
    print(f"MSE Loss:     {mse_ssim_mean:.3f} ± {np.std(results['mse_ssim']):.3f} (+{mse_vs_zf_ssim:.3f})")
    print(f"⊥-loss:       {perp_ssim_mean:.3f} ± {np.std(results['perp_ssim']):.3f} (+{perp_vs_zf_ssim:.3f})")
    print(f"")
    print(f"⊥-LOSS ADVANTAGE:")
    print(f"PSNR: +{perp_vs_mse_psnr:.2f} dB (p={pvalue_psnr:.4f})")
    print(f"SSIM: +{perp_vs_mse_ssim:.3f} (p={pvalue_ssim:.4f})")
    print(f"Win Rate: {perp_wins_psnr}/{len(slice_indices)} PSNR ({100*perp_wins_psnr/len(slice_indices):.1f}%)")
    print(f"Win Rate: {perp_wins_ssim}/{len(slice_indices)} SSIM ({100*perp_wins_ssim/len(slice_indices):.1f}%)")
    
    significance_psnr = "SIGNIFICANT" if pvalue_psnr < 0.05 else "Not significant"
    significance_ssim = "SIGNIFICANT" if pvalue_ssim < 0.05 else "Not significant"
    print(f"Statistical significance: PSNR {significance_psnr}, SSIM {significance_ssim}")

    # CREATE BEAUTIFUL COMPARISON PLOTS
    create_comparison_plots(results, run_dir, args)

    # Save detailed results to file
    results_file = os.path.join(run_dir, 'detailed_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=== MSE vs ⊥-LOSS COMPARISON RESULTS ===\n")
        f.write(f"MSE Model: {args.mse_model}\n")
        f.write(f"⊥-loss Model: {args.perp_model}\n")
        f.write(f"Slices: {slice_indices}\n")
        f.write(f"Acceleration: {args.acceleration}x\n")
        f.write(f"Random seed: {args.seed}\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write(f"⊥-loss PSNR advantage: +{perp_vs_mse_psnr:.3f} dB (p={pvalue_psnr:.4f})\n")
        f.write(f"⊥-loss SSIM advantage: +{perp_vs_mse_ssim:.4f} (p={pvalue_ssim:.4f})\n")
        f.write(f"⊥-loss win rate: {perp_wins_psnr}/{len(slice_indices)} ({100*perp_wins_psnr/len(slice_indices):.1f}%)\n\n")
        
        f.write("Slice\tZF_PSNR\tZF_SSIM\tMSE_PSNR\tMSE_SSIM\tPERP_PSNR\tPERP_SSIM\tAdvantage\n")
        for i, slice_idx in enumerate(slice_indices):
            advantage = results['perp_psnr'][i] - results['mse_psnr'][i]
            f.write(f"{slice_idx}\t{results['zf_psnr'][i]:.3f}\t{results['zf_ssim'][i]:.4f}\t")
            f.write(f"{results['mse_psnr'][i]:.3f}\t{results['mse_ssim'][i]:.4f}\t")
            f.write(f"{results['perp_psnr'][i]:.3f}\t{results['perp_ssim'][i]:.4f}\t{advantage:.3f}\n")
    
    print(f"\nComparison completed successfully!")
    print(f"Results saved in: {run_dir}")
    print(f"📊 Detailed results: {results_file}")
    print(f"🖼️  Individual slice comparisons: {len(saved_slices)} PNG files")
    print(f"📈 Comparison plots: multiple PNG files")
    
    if perp_vs_mse_psnr > 0:
        print(f"🎉 ⊥-loss WINS! +{perp_vs_mse_psnr:.2f}dB improvement over MSE loss!")
    else:
        print(f"📉 MSE performs better by {-perp_vs_mse_psnr:.2f}dB")

if __name__ == "__main__":
    main()