# Advanced Machine Learning & Signal Processing: Deep Equilibrium Architectures for Inverse Problems in Signal Reconstruction

**Master's Thesis**
**Deep Equilibrium solver for ill-posed inverse problems, leveraging signal sparsity and implicit differentiation for memory-efficient reconstruction.**

## Overview

**About thesis**


My thesis explores the intersection of classical optimization theory and modern deep learning for solving inverse problems in signal reconstruction. By leveraging Deep Equilibrium Models (DEQs), spectral normalization techniques, symmetric loss functions and sparsity theory, this work addresses critical challenges in compressed sensing , focusing on MRI reconstruction images.

---


### The Problem We're Solving
 **How do we reconstruct high-fidelity signals from severely undersampled or corrupted measurements?**

Traditional approaches require sampling at twice the signal's bandwidth (Nyquist-Shannon theorem), but modern applications demand real-time processing with limited bandwidth, storage, and power constraints. This thesis tackles this problem through:

- **Compressed Sensing** - Exploiting signal sparsity to reconstruct from 4-8x fewer samples
- **Deep Equilibrium Models** - Infinite-depth networks with constant memory footprint
- **Fixed-Point Theory** - Mathematically rigorous convergence guarantees
- **Perpendicular Loss Functions** - Symmetric error distributions for complex-valued signals

**Signal processing concepts applied:** Fourier domain undersampling, k-space geometry, aliasing artifact formation, and the difference between k-space cropping (low-pass filtering) vs. image-space cropping.
---

## Key Contributions & Technical Innovation


###  Multi-Slice Volumetric Data Processing Pipeline

Developed a custom PyTorch Dataset class (`MultiSliceFastMRIDataloader`) that processes 3D multi-channel signal volumes efficiently. Unlike single-slice approaches, this pipeline:

- Extracts anatomically-rich slices (indices 11-27) from volumetric data
- Handles multi-coil k-space data with coil sensitivity map estimation
- Performs center cropping in k-space vs. image domain (critical for aliasing artifacts)
- Implements instance normalization with outlier clipping for stable training


### Comprehensive Validation and Ablation Framework

Created extensive validation pipelines comparing different loss functions, solver configurations, and denoising architectures. All validation scripts, comparison tools, and visualization utilities are original contributions that enabled rigorous evaluation of the proposed methods.

**Includes:** Multi-metric evaluation tools (PSNR, SSIM, VIF), noise robustness testing, computational profiling, and visual quality assessment frameworks.
- Compatible with **GroupNorm** for batch-independent inference

### Implementation of Deep Equilibrium Models for Signal Reconstruction

Building upon the theoretical framework from Gilton et al. (2021), I implemented and optimized Deep Equilibrium architectures that bypass the fundamental memory-depth tradeoff in traditional unrolled networks. The key insight: instead of fixing the number of iterations during training (typical unrolled nets use 5-10), we solve for the fixed point directly using implicit differentiation.

**What I implemented:**
- Proximal gradient-based DEQ solver with Anderson acceleration
- Implicit backpropagation through fixed-point equations (constant O(1) memory vs. O(K) for unrolled)
- Adaptive iteration budgets at inference time (critical for edge deployment)
- Integration with modern forward models (Fourier operators, k-space undersampling)



### Integration of Symmetric Perpendicular Loss for Complex Signal Reconstruction

Standard L2 loss in the complex domain exhibits an asymmetric loss landscape, systematically biasing reconstructions toward magnitude underestimation. Following Terpstra et al.'s (2022) perpendicular loss formulation, I implemented the symmetric ⊥-loss that decouples magnitude and phase errors via scalar rejection.

**Mathematical foundation:**
For complex signals Y, Ŷ ∈ ℂ^(m×n), the perpendicular component is:

⊥(Ŷ, Y) = |Im(Ŷ)·Re(Y) - Re(Ŷ)·Im(Y)| / |Ŷ|

Combined with magnitude loss f(|Y|, |Ŷ|), this creates a symmetric error surface crucial for unbiased reconstruction.

**My implementation:** Full training pipeline with ⊥-loss, comparison against MSE/SSIM baselines, and validation across multiple noise levels. This loss function generalizes beautifully to vector field regression (R²) for motion estimation tasks.

### Spectral-Normalized DnCNN as Lipschitz-Constrained Denoiser

To guarantee convergence of the equilibrium solver, the regularizer network must satisfy Lipschitz continuity. I designed and trained a DnCNN architecture with spectral normalization on all convolutional layers, ensuring ||∇R(x)|| ≤ 1.

**Architecture details:**
- 17-layer residual CNN with spectral normalization (Chen et al.)
- GroupNorm layers for stable batch-independent inference
- Residual learning: R(x) = I + N(x), where N is the artifact estimator
- 64 feature channels, 3×3 kernels throughout

**Concept integration:** This connects classical operator theory (contractive mappings) with modern deep learning (CNNs), enabling provable convergence while maintaining expressive power.

---

## 🏗️ Architecture & Components

### Core Models

#### **Deep Equilibrium Proximal Gradient (DE-Prox)**
```
Input: Undersampled measurements y
Forward Model: A (e.g., FFT with undersampling mask)
Iteration: x(t+1) = DnCNN(x(t) + A*(y - Ax(t)))
Output: x* (fixed point solution)
```

#### **DnCNN Denoiser (Spectral-Normalized)**
```python
- 17-layer Residual CNN
- Spectral normalization on all conv layers (Lipschitz constraint)
- GroupNorm for batch-independent processing
- 64 feature channels
- 3×3 kernels with padding=1
```

### Data Processing Pipeline

```
Multi-Slice Volume (H×W×D)
    ↓
Slice Selection (indices 11-27, anatomically rich)
    ↓
K-Space Undersampling (4×-8× acceleration)
    ↓
Coil Sensitivity Estimation (U-Net based)
    ↓
Complex Image Formation (320×320 patches)
    ↓
Instance Normalization (mean=0, std=1, clamp [-6,6])
    ↓
Training Pairs (input_undersampled, target_full)
```

---

## Experimental Design & Validation

The thesis includes extensive experimental validation across multiple configurations, loss functions, and noise levels. Comprehensive results, ablation studies, and comparative analysis are documented in the full thesis report:

**[Efarmogh-Methodwn-Deep-Equilibrium-Se-Problhmata-Antistrofhs-Apeikonisis.pdf](Efarmogh-Methodwn-Deep-Equilibrium-Se-Problhmata-Antistrofhs-Apeikonisis.pdf)** (100+ pages)


---

## Installation & Setup

### Prerequisites
```
Python 3.8+
PyTorch 1.9+
CUDA 11.1+ (for GPU acceleration)
```

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/deep-equilibrium-inverse.git
cd deep-equilibrium-inverse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Download sample dataset (FastMRI knee dataset)
# Register at https://fastmri.med.nyu.edu/
```

### Training Example
```bash
# Train DE-Prox with perpendicular loss (4× undersampling)
python deep_equilibrium_inverse/scripts/fixedpoint/mri_prox_fixedeta_pre_and_perp.py \
    --data_path /path/to/fastmri \
    --batch_size 4 \
    --num_epochs 50 \
    --acceleration 4 \
    --use_perp_loss

# Validate trained model
python deep_equilibrium_inverse/scripts/validation/validate_deq_mri.py \
    --checkpoint checkpoints/best_model.pth \
    --output_dir validation_results
```

### Custom Dataset Integration
```python
from deep_equilibrium_inverse.utils.fastmri_dataloader import MultiSliceFastMRIDataloader

# Load your multi-channel signal data
dataset = MultiSliceFastMRIDataloader(
    dataset_location='/path/to/data',
    data_indices=range(1000),  # Use subset for quick testing
    sketchynormalize=True
)

# Integrate with your training loop
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

---

## Theoretical Foundation

### Papers Integrated

**1. Deep Equilibrium Architectures for Inverse Problems in Imaging**  
*Davis Gilton, Gregory Ongie, Rebecca Willett (2021)*  
- Introduced DEQ framework for imaging inverse problems
- Proved convergence guarantees for contractive operators
- Demonstrated implicit differentiation for memory efficiency

**2. ⊥-loss: A Symmetric Loss Function for Complex-Valued Reconstruction**  
*Maarten L. Terpstra et al. (2022)*  
- Identified magnitude bias in standard L2 complex loss
- Proposed phase-magnitude decoupling via scalar rejection
- Showed 10-15% improvement in reconstruction metrics

---

## Signal Processing Concepts Applied

Throughout this thesis, I integrated fundamental signal processing and optimization theory with modern deep learning:

**Compressed Sensing Theory**
- Restricted Isometry Property (RIP) for measurement matrices
- Incoherence between sensing and sparsity bases
- L1 minimization vs. learned regularizers
- Recovery guarantees for sparse signals

**Fourier Analysis & K-Space Processing**
- 2D FFT/iFFT for image-frequency domain conversion
- Undersampling patterns (Cartesian, radial, Poisson disc)
- Aliasing artifacts and their relationship to k-space coverage
- Conjugate symmetry in k-space for real-valued signals

---

## Future Directions & Extensions

The techniques developed here open several promising research directions:

**Architectural Extensions**
- Full 3D volumetric processing (current implementation processes 2D slices)
- Multi-scale equilibrium models with hierarchical fixed-points
- Learnable step-sizes and iteration-dependent regularizers
- Uncertainty quantification through Bayesian formulations

**Application Domains**
- Weather prediction from sparse meteorological sensors
- Seismic imaging for geophysical exploration
- Financial time-series reconstruction and forecasting
- Power grid state estimation from phasor measurements
- Video frame interpolation and super-resolution

---

## References & Citation

If you find this work useful for your research, please cite the foundational papers:

```bibtex
@article{Gilton2021,
  title={Deep Equilibrium Architectures for Inverse Problems in Imaging},
  author={Gilton, Davis and Ongie, Gregory and Willett, Rebecca},
  journal={IEEE Transactions on Computational Imaging},
  year={2021},
  note={Base DEQ framework and convergence theory}
}

@article{Terpstra2022,
  title={⊥-loss: A symmetric loss function for magnetic resonance imaging reconstruction},
  author={Terpstra, Maarten L and Maspero, Matteo and Sbrizzi, Alessandro and van den Berg, Cornelis AT},
  journal={Medical Image Analysis},
  volume={80},
  pages={102509},
  year={2022},
  note={Perpendicular loss formulation}
}

```

---

## About This Work

**Author:** Andreas Katsaros  
**Student ID:** 1084522  
**Institution:** University of Patras, Department of Electrical and Computer Engineering  
**Thesis Title:** Εφαρμογή Μεθόδων Deep Equilibrium σε Προβλήματα Αντίστροφης Απεικόνισης  
**Grade Achieved:** 10/10  
**Year:** 2025

**Full Thesis Document:** [Efarmogh-Methodwn-Deep-Equilibrium-Se-Problhmata-Antistrofhs-Apeikonisis.pdf](Efarmogh-Methodwn-Deep-Equilibrium-Se-Problhmata-Antistrofhs-Apeikonisis.pdf)

https://nemertes.library.upatras.gr/search?spc.page=1&query=EQUILIBRIUM%20%CE%9A%CE%91%CE%A4%CE%A3%CE%91%CE%A1%CE%9F%CE%A3 

This thesis represents comprehensive work at the intersection of classical optimization, modern deep learning, and signal processing theory. The implementation demonstrates mastery of advanced concepts in compressed sensing, fixed-point theory, complex analysis, and deep neural networks, with direct applications to telecommunications, computer vision, and industrial automation.

This work builds upon open-source research from the signal processing and machine learning communities. The base DEQ framework code is attributed to Gilton et al. (2021), with my contributions focused on implementation, optimization, loss function integration, and extensive experimental validation.

---




## Code Attribution & Legal Notice

This repository is based on the Deep Equilibrium framework by Gilton et al. (2021).
The original codebase (networks/, operators/, solvers/, training/) is their work from which i was highly influenced.

**Original Repository:** https://github.com/dgilton/deep_equilibrium_inverse
**Paper:** "Deep Equilibrium Architectures for Inverse Problems in Imaging"

**Contributions (for thesis evaluation):**
- Perpendicular loss integration (scripts/fixedpoint/mri_prox_fixedeta_pre_and_perp.py)
- Multi-slice dataloader (utils/fastmri_dataloader.py)
- DnCNN denoiser implementation (scripts/denoising/)
- Complete validation framework (scripts/validation/)

**Status:** The original codebase does not include an explicit license. 
This repository is uploaded for academic portfolio purposes only.
If you are the original author and would like this removed, please contact me.

##  Acknowledgments

- **FastMRI Challenge Team** - NYU Langone Health for open dataset
- **Deep Equilibrium Models Community** - Foundational DEQ research
- **PyTorch Team** - Exceptional deep learning framework
- **Thesis Supervisor** - Invaluable guidance and feedback
  
---
