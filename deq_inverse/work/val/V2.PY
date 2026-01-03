#!/usr/bin/env python
# validate_deq.py
import os
import sys
import argparse
import random
import time
import math

import torch
import torch.nn as nn
import numpy as np

# Adjust this path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, root_dir)

# repo imports (same used in training)
import operators.singlecoil_mri as mrimodel
from operators.operator import OperatorPlusNoise
from utils.fastmri_dataloader import MultiSliceFastMRIDataloader
from networks.normalized_equilibrium_u_net import DnCNN
from solvers.equilibrium_solvers import EquilibriumProxGradMRI
from solvers import new_equilibrium_utils as eq_utils

# optional SSIM
try:
    from skimage.metrics import structural_similarity as sk_ssim
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

from PIL import Image
import matplotlib.pyplot as plt

def load_checkpoint_to_solver(solver, ckpt_path, device):
    """
    Loads solver_state_dict (or state_dict) from checkpoint into solver.
    Handles 'module.' prefixes and optional 'eta' top-level saved item.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    chk = torch.load(ckpt_path, map_location=device)

    # checkpoint may have different layouts
    if isinstance(chk, dict):
        if 'solver_state_dict' in chk:
            state_dict = chk['solver_state_dict']
        elif 'state_dict' in chk:
            state_dict = chk['state_dict']
        else:
            # assume entire dict is a state_dict
            state_dict = chk
    else:
        state_dict = chk

    # If saved state_dict contains 'eta' as a top-level param inside the dict,
    # remove it and set solver.eta if solver has that attribute.
    if isinstance(state_dict, dict) and 'eta' in state_dict:
        try:
            eta_value = state_dict.pop('eta')
            if hasattr(solver, 'eta'):
                # solver might be DataParallel; handle both
                tsolver = getattr(solver, 'module', solver)
                try:
                    tsolver.eta.data = torch.tensor(eta_value, device=tsolver.eta.device)
                except Exception:
                    try:
                        tsolver.eta = torch.tensor(eta_value, device=device)
                    except Exception:
                        pass
        except Exception:
            pass

    # Clean up module prefixes if necessary
    def _clean_state_dict(sd):
        new_sd = {}
        for k, v in sd.items():
            # handle keys starting with "module."
            if k.startswith('module.'):
                new_sd[k[len('module.'):]] = v
            else:
                new_sd[k] = v
        return new_sd

    try:
        cleaned = _clean_state_dict(state_dict)
        solver.load_state_dict(cleaned, strict=False)
        print("Loaded solver_state_dict into solver.")
    except Exception as e:
        print("Warning: loading state dict failed with strict=False; trying alternative loading.", e)
        try:
            solver.load_state_dict(state_dict, strict=False)
            print("Loaded state dict into solver (fallback).")
        except Exception as e2:
            print("Failed to load checkpoint into solver.", e2)
            raise e2

    return chk  # return original checkpoint dict for any extra metadata


def mag_from_complex_tensor(x):
    # x shape [B, 2, H, W]
    return torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2)


def psnr_per_slice(recon_mag, target_mag, data_range=None):
    # recon_mag, target_mag: numpy arrays HxW each
    mse = np.mean((recon_mag - target_mag) ** 2)
    if data_range is None:
        dr = float(target_mag.max() - target_mag.min())
        if dr <= 0:
            dr = float(np.max(target_mag) if np.max(target_mag) > 0 else 1.0)
    else:
        dr = float(data_range)
    if mse <= 0:
        return float('inf')
    return 10.0 * math.log10((dr ** 2) / mse)


def compute_ssim(recon_mag, target_mag):
    if not HAVE_SKIMAGE:
        return float('nan')
    # skimage expects images in range [min, max] and 2D
    try:
        s = sk_ssim(target_mag, recon_mag, data_range=target_mag.max() - target_mag.min())
        return float(s)
    except Exception:
        return float('nan')


def save_image_pair(recon_mag_np, target_mag_np, outpath, idx):
    # Normalize to 0..255 for saving (per image)
    eps = 1e-12
    mn = min(recon_mag_np.min(), target_mag_np.min())
    mx = max(recon_mag_np.max(), target_mag_np.max())
    rng = max(mx - mn, eps)
    r = (recon_mag_np - mn) / rng
    t = (target_mag_np - mn) / rng
    r8 = (np.clip(r, 0, 1) * 255.0).astype(np.uint8)
    t8 = (np.clip(t, 0, 1) * 255.0).astype(np.uint8)
    # stack them side by side
    h, w = r8.shape
    canvas = np.zeros((h, w*2), dtype=np.uint8)
    canvas[:, :w] = t8
    canvas[:, w:] = r8
    img = Image.fromarray(canvas)
    fname = os.path.join(outpath, f"val_pair_{idx:04d}.png")
    img.save(fname)
    return fname


def main():
    parser = argparse.ArgumentParser(description="Validate DEQ checkpoint on MultiSliceFastMRI data")
    parser.add_argument('--ckpt', type=str, default=r"G:\AK\deq_model\deq_PROX_fixedeta_pre_and4_final_ritsa_FINAL_22.ckpt",
                        help='Path to checkpoint to validate')
    parser.add_argument('--data', type=str, default=r"G:\AK\data\singlecoil_train",
                        help='Path to dataset directory (H5 files)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_val_slices', type=int, default=10, help='How many validation slices to evaluate (0=all)')
    parser.add_argument('--out', type=str, default='./val_outputs', help='Output directory for images + logs')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    print("Validation device:", device)

    # build mask (same as training)
    dataheight = 320
    datawidth = 320
    mri_center_fraction = 0.04
    mri_acceleration = 8.0
    mask = mrimodel.create_mask(shape=[dataheight, datawidth, 2], acceleration=mri_acceleration,
                                center_fraction=mri_center_fraction, seed=10)

    forward_operator = mrimodel.cartesianSingleCoilMRI(kspace_mask=mask).to(device=device)
    measurement_process = OperatorPlusNoise(forward_operator, noise_sigma=1e-2).to(device=device)
    internal_forward_operator = mrimodel.cartesianSingleCoilMRI(kspace_mask=mask).to(device=device)

    # load model architecture (same as training)
    n_channels = 2
    learned_component = DnCNN(n_channels, num_of_layers=17, lip=1.0)

    # build solver/DEQ wrapper
    solver = EquilibriumProxGradMRI(linear_operator=internal_forward_operator,
                                    nonlinear_operator=learned_component,
                                    eta=0.2, minval=-6, maxval=6)

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print("Using DataParallel on GPUs")
        solver = nn.DataParallel(solver)

    solver = solver.to(device)
    solver.eval()

    # Simple validation function to avoid DEQ gradient hook issues
    def simple_validation_forward(y, initial_point, max_iter=10, tol=1e-4):
        """Simple forward iteration for validation without DEQ hooks"""
        x = initial_point
        for i in range(max_iter):
            x_new = solver(x, y)  # solver(z, x) where z=current, x=measurement
            residual = (x_new - x).norm().item() / (1e-7 + x_new.norm().item())
            x = x_new
            if residual < tol:
                break
        return x

    # Load checkpoint
    print("Loading checkpoint:", args.ckpt)
    chk = load_checkpoint_to_solver(solver, args.ckpt, device)
    # optionally inspect checkpoint keys:
    if isinstance(chk, dict):
        print("Checkpoint keys:", list(chk.keys()))

    # Build validation dataset (use deterministic small subset)
    dataset = MultiSliceFastMRIDataloader(args.data, data_indices=None)
    total_slices = len(dataset)
    print(f"Total slices in dataset: {total_slices}")

    # prepare indices for validation: pick evenly spaced slices (or first N)
    if args.num_val_slices <= 0 or args.num_val_slices > total_slices:
        val_indices = list(range(total_slices))
    else:
        step = max(1, total_slices // args.num_val_slices)
        val_indices = list(range(0, total_slices, step))[:args.num_val_slices]

    print(f"Validating on {len(val_indices)} slices (indices sample)")

    val_dataset = MultiSliceFastMRIDataloader(args.data, data_indices=val_indices)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    os.makedirs(args.out, exist_ok=True)

    per_slice_psnrs = []
    per_slice_ssims = []
    saved_count = 0
    t0 = time.time()

    solver.eval()  # Set model to evaluation mode
    
    # Use simple forward iteration for validation (no backward hooks)
    for bidx, batch in enumerate(val_loader):
        with torch.no_grad():  # No gradients needed for validation
            input_img, target_img = batch  # both [B, 2, H, W]
            input_img = input_img.to(device=device)
            target_img = target_img.to(device=device)

            # measurement (simulate noisy measurement around target)
            y = measurement_process(target_img)

            # initial point for DEQ
            initial_point = forward_operator.adjoint(y)

            # run DEQ forward (using simple iteration, no hooks)
            recon = simple_validation_forward(y, initial_point=initial_point)  # recon [B,2,H,W]
            recon = torch.clamp(recon, -6, 6)

            # compute magnitudes
            recon_mag = mag_from_complex_tensor(recon).cpu().numpy()  # shape [B, H, W]
            target_mag = mag_from_complex_tensor(target_img).cpu().numpy()

            B = recon_mag.shape[0]
            for i in range(B):
                r = recon_mag[i]
                timg = target_mag[i]
                # data range for PSNR: use target span
                dr = float(timg.max() - timg.min())
                if dr <= 1e-12:
                    dr = 1.0
                ps = psnr_per_slice(r, timg, data_range=dr)
                per_slice_psnrs.append(ps)

                s = compute_ssim(r, timg) if HAVE_SKIMAGE else float('nan')
                per_slice_ssims.append(s)

                # save a few image pairs to disk (target | recon)
                if saved_count < 20:
                    fname = save_image_pair(r, timg, args.out, saved_count)
                    saved_count += 1

            if (bidx + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"Processed { (bidx+1)*args.batch_size } slices in {elapsed:.1f}s")

    t_total = time.time() - t0
    per_slice_psnrs = np.array(per_slice_psnrs)
    per_slice_ssims = np.array(per_slice_ssims)

    print("Validation finished.")
    print(f"Processed slices: {len(per_slice_psnrs)} in {t_total:.1f}s")
    print(f"PSNR mean: {np.nanmean(per_slice_psnrs):.3f} dB, std: {np.nanstd(per_slice_psnrs):.3f} dB")
    if HAVE_SKIMAGE:
        print(f"SSIM mean: {np.nanmean(per_slice_ssims):.4f}, std: {np.nanstd(per_slice_ssims):.4f}")
    else:
        print("skimage not installed; SSIM skipped (install scikit-image to enable)")

    # save numeric summary
    summary_path = os.path.join(args.out, "validation_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"checkpoint: {args.ckpt}\n")
        f.write(f"slices: {len(per_slice_psnrs)}\n")
        f.write(f"PSNR mean: {np.nanmean(per_slice_psnrs):.3f}\n")
        f.write(f"PSNR std: {np.nanstd(per_slice_psnrs):.3f}\n")
        if HAVE_SKIMAGE:
            f.write(f"SSIM mean: {np.nanmean(per_slice_ssims):.4f}\n")
            f.write(f"SSIM std: {np.nanstd(per_slice_ssims):.4f}\n")
        f.write(f"total_time_s: {t_total:.1f}\n")
    print(f"Saved summary to {summary_path}")
    print(f"Saved example pairs to {args.out}")

if __name__ == "__main__":
    main()
