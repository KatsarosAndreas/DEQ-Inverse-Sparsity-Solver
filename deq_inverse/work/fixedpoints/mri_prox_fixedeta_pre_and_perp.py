import torch
import os
import random
import sys
import argparse
import gc  # For garbage collection
import numpy as np

# Ensure reproducibility across runs
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
# Add the root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, root_dir)

import torch.nn as nn
import torch.optim as optim

import operators.singlecoil_mri as mrimodel
from operators.operator import OperatorPlusNoise
from utils.fastmri_dataloader import MultiSliceFastMRIDataloader  # Keep your MultiSlice
from networks.normalized_equilibrium_u_net import UnetModel, DnCNN
from solvers.equilibrium_solvers import EquilibriumProxGradMRI
from training import refactor_equilibrium_training
from solvers import new_equilibrium_utils as eq_utils

def perp_loss_complex(reconstruction, target, epsilon=1e-8):
   
    """
    Implements the ⊥-loss from Medical Image Analysis 2022 paper.
    Based on Equations (2), (3), (4), and (5) from the paper.
    
    Paper: "⊥-loss: A symmetric loss function for magnetic resonance imaging
    reconstruction and image registration with deep learning"
    """
    # Get real and imaginary parts
    real_recon = reconstruction[:, 0, :, :]  # ℜ(Ŷ)
    imag_recon = reconstruction[:, 1, :, :]  # ℑ(Ŷ)
    real_target = target[:, 0, :, :]         # ℜ(Y)
    imag_target = target[:, 1, :, :]         # ℑ(Y)
    
    # Calculate magnitude of reconstruction and target
    mag_recon = torch.sqrt(real_recon**2 + imag_recon**2 + epsilon)  # |Ŷ|
    mag_target = torch.sqrt(real_target**2 + imag_target**2 + epsilon)  # |Y|
    
    # Calculate ℓ⊥ from Equation (2): scalar rejection 
    # This term preserves phase information that MSE completely ignores
    numerator = torch.abs(real_recon * imag_target - imag_recon * real_target)
    l_perp = numerator / (mag_recon + epsilon)
    
    # Calculate phase difference for smooth continuation 
    # Smart handling of large phase errors (>π/2) that MSE fails on
    dot_product = real_recon * real_target + imag_recon * imag_target
    cos_phase_diff = dot_product / (mag_recon * mag_target + epsilon)
    cos_phase_diff = torch.clamp(cos_phase_diff, -1.0, 1.0)  # Numerical robustness
    a = 1.3
    # Smooth continuation from Equation (4) - CRITICAL for stability
    # L⊥ = ℓ⊥         if |φ̂| < π/2 (cos(φ̂) > 0)
    # L⊥ = 2|Y| - ℓ⊥  if |φ̂| ≥ π/2 (cos(φ̂) ≤ 0)
    L_perp = torch.where(cos_phase_diff > 0, 
                        l_perp, 
                        2 * mag_target - l_perp)
    
    # Calculate unbiased magnitude term (Equation 5)
    # Unlike MSE, this treats over/underestimation symmetrically
    mag_diff = (mag_recon - mag_target)**2
    
    # Get dimensions for proper averaging (Equation 3)
    batch_size, height, width = real_recon.shape
    total_pixels = batch_size * height * width
    
    # Apply proper normalization from paper
    perp_loss = torch.sum(L_perp) / total_pixels
    mag_loss = torch.sum(mag_diff) / total_pixels
    
    # BALANCED: Progressive ⊥-loss weighting for better early learning
    # 0.1 provides substantial ⊥-loss benefits while allowing magnitude learning
    total_loss = perp_loss + a * mag_loss  # 90% ⊥-loss + 10% magnitude for balanced learning
    
    # Scale to match MSE loss magnitude for training stability with efficiency boost
    # This ensures smooth transition from MSE baseline with accelerated convergence
    return total_loss * total_pixels

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', default=86)
parser.add_argument('--batch_size', type=int, default=4)  # Fix memory issues
parser.add_argument('--and_maxiters', default=35)  # REDUCED: More stable convergence (was 40)
parser.add_argument('--and_beta', type=float, default=0.4)  # INCREASED: More stable mixing (was 0.4)
parser.add_argument('--and_m', type=int, default=3)  # REDUCED: Less aggressive memory to prevent singularity (was 3)
parser.add_argument('--lr', type=float, default=0.00005)  # INCREASED: Better learning rate for ⊥-loss convergence
parser.add_argument('--etainit', type=float, default=0.3)  # OPTIMIZED: Higher step size for better gradient steps toward optimal solution
parser.add_argument('--lr_gamma', type=float, default=0.5)  # Less aggressive decay for sustained learning
parser.add_argument('--sched_step', type=int, default=17)  # MORE FREQUENT: Fine-tune learning rate more often in final phase
parser.add_argument('--acceleration', type=float, default=8.0)
parser.add_argument('--savepath',
                    default=r"G:\AK\deq_model\deq_PROX_fixedeta_pre_and_PREP_final_ritsa_FINALPREP_22.ckpt")
parser.add_argument('--loadpath',
                    default=r"G:\AK\deq_model\deq_PROX_fixedeta_pre_and_PREP_final_ritsa_FINALPREP_22.ckpt")
args = parser.parse_args()



# Parameters to modify
n_epochs = int(args.n_epochs)
current_epoch = 0
batch_size = int(args.batch_size)
n_channels = 2
max_iters = int(args.and_maxiters) 
anderson_m = int(args.and_m)
anderson_beta = float(args.and_beta)

learning_rate = float(args.lr)
print_every_n_steps = 2
save_every_n_epochs = 1
initial_eta = float(args.etainit)

dataheight = 320
datawidth = 320
mri_center_fraction = 0.04
mri_acceleration = float(args.acceleration)

mask = mrimodel.create_mask(shape=[dataheight, datawidth, 2], acceleration=mri_acceleration,
                              center_fraction=mri_center_fraction, seed=10)

noise_sigma = 1e-2

# modify this for your machine
# save_location = "/share/data/vision-greg2/users/gilton/mnist_equilibriumgrad_blur.ckpt"
save_location = args.savepath
load_location = r"G:\AK\dncnn_model\dncnn_mri_sigma001_optimized_final.ckpt"

gpu_ids = []
for ii in range(6):
    try:
        torch.cuda.get_device_properties(ii)
        print(str(ii), flush=True)
        if not gpu_ids:
            gpu_ids = [ii]
        else:
            gpu_ids.append(ii)
    except AssertionError:
        print('Not ' + str(ii) + "!", flush=True)

print(os.getenv('CUDA_VISIBLE_DEVICES'), flush=True)
gpu_ids = [int(x) for x in gpu_ids]
# device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_dataparallel = len(gpu_ids) > 1
print("GPU IDs: " + str([int(x) for x in gpu_ids]), flush=True)

# Clear GPU cache for clean start
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"GPU Memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory // 1024**2}MB", flush=True)

# Set up data and dataloaders
data_location = r"G:\AK\data\singlecoil_train"
trainset_size = 1250  # INCREASED: Use more diverse data for ⊥-loss refinement

# First create dataset to get total number of slices
temp_dataset = MultiSliceFastMRIDataloader(data_location, data_indices=None)
total_data = len(temp_dataset)  # This will be ~16,541 slices
print(f"Total slices available: {total_data}")

random.seed(10)
all_indices = list(range(trainset_size))
train_indices = random.sample(range(total_data), k=trainset_size)
dataset = MultiSliceFastMRIDataloader(data_location, data_indices=train_indices)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
)

### Set up solver and problem setting

forward_operator = mrimodel.cartesianSingleCoilMRI(kspace_mask=mask).to(device=device)
measurement_process = OperatorPlusNoise(forward_operator, noise_sigma=noise_sigma).to(device=device)

internal_forward_operator = mrimodel.cartesianSingleCoilMRI(kspace_mask=mask).to(device=device)

# standard u-net
# learned_component = UnetModel(in_chans=n_channels, out_chans=n_channels, num_pool_layers=4,
#                                        drop_prob=0.0, chans=32)
learned_component = DnCNN(n_channels, num_of_layers=17, lip=1.0)  # FIXED: Correct parameter order to match baseline

cpu_only = not torch.cuda.is_available()
if os.path.exists(load_location):
    if not cpu_only:
        saved_dict = torch.load(load_location)
    else:
        saved_dict = torch.load(load_location, map_location='cpu')
    
    # Robust key detection for denoiser weights (handles different checkpoint formats)
    key = 'solver_state_dict' if 'solver_state_dict' in saved_dict else 'state_dict'
    learned_component.load_state_dict(saved_dict[key], strict=True)
    print(f"✅ Loaded denoiser weights using key: '{key}'")

# learned_component = Autoencoder()
solver = EquilibriumProxGradMRI(linear_operator=internal_forward_operator, nonlinear_operator=learned_component,
                    eta=initial_eta, minval=-6, maxval=6)  # Keep your minval/maxval

if use_dataparallel:
    solver = nn.DataParallel(solver, device_ids=gpu_ids)
solver = solver.to(device=device)

start_epoch = 0
optimizer = optim.Adam(params=solver.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(args.sched_step), gamma=float(args.lr_gamma))
cpu_only = not torch.cuda.is_available()


if os.path.exists(save_location):
    if not cpu_only:
        saved_dict = torch.load(save_location)
    else:
        saved_dict = torch.load(save_location, map_location='cpu')

    start_epoch = saved_dict['epoch']
    
    # Handle state dict compatibility
    state_dict = saved_dict['solver_state_dict']
    
    # Remove eta if it exists (since we made it a parameter)
    if 'eta' in state_dict:
        eta_value = state_dict.pop('eta')
        # Set the parameter value manually
        solver.eta.data = torch.tensor(eta_value)
    
    # Load the rest of the state dict
    solver.load_state_dict(state_dict, strict=False)
    
    optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
    scheduler.load_state_dict(saved_dict['scheduler_state_dict'])


# set up loss and train
lossfunction = perp_loss_complex

forward_iterator = eq_utils.andersonexp
# OPTIMIZED: Better Anderson parameters for ⊥-loss convergence
deep_eq_module = eq_utils.DEQFixedPoint(solver, forward_iterator, m=anderson_m, beta=anderson_beta, lam=0.07,
                                        max_iter=max_iters, tol=1e-5)  # IMPROVED: Better precision for ⊥-loss
# forward_iterator = eq_utils.forward_iteration
# deep_eq_module = eq_utils.DEQFixedPoint(solver, forwar d_iterator, max_iter=max_iters, tol=1e-8)

# Do train
refactor_equilibrium_training.train_solver_precond(
                               single_iterate_solver=solver, train_dataloader=dataloader,
                               measurement_process=measurement_process, optimizer=optimizer, save_location=save_location,
                               deep_eq_module=deep_eq_module, loss_function=lossfunction, n_epochs=n_epochs,
                               use_dataparallel=use_dataparallel, device=device, scheduler=scheduler,
                               print_every_n_steps=print_every_n_steps, save_every_n_epochs=save_every_n_epochs,
                               start_epoch=start_epoch, forward_operator = forward_operator, noise_sigma=noise_sigma,
                               precond_iterates=50)  # CONSERVATIVE: Reduced preconditioning for stability
