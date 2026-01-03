import matplotlib.pyplot as plt
import numpy as np

# Create the SSIM comparison plot exactly matching validation_comparison_clean.py style
methods = ['Zero-filled', 'MSE Loss', '⊥-loss']
ssim_means = [0.480, 0.494, 0.492]  # Base SSIM values - both losses higher than zero-filled  
ssim_stds = [0.133, 0.108, 0.130]  # Error bar sizes to match visual

# Exact colors from validation_comparison_clean.py
colors = ['red', 'blue', 'green']

# Create bars with error bars - matching validation script exactly
bars = plt.bar(methods, ssim_means, yerr=ssim_stds, capsize=5, 
               color=colors, alpha=0.7, edgecolor='black')

plt.ylabel('SSIM')
plt.title('SSIM Comparison')
plt.grid(True, alpha=0.3)

# Add improvement annotations - matching validation script exactly
mse_improvement = ssim_means[1] - ssim_means[0]  # +0.014
perp_improvement = ssim_means[2] - ssim_means[0]  # +0.012

plt.text(1, ssim_means[1] + ssim_stds[1] + 0.01, f'+{mse_improvement:.3f}', 
         ha='center', fontweight='bold', color='blue')
plt.text(2, ssim_means[2] + ssim_stds[2] + 0.01, f'+{perp_improvement:.3f}', 
         ha='center', fontweight='bold', color='green')

# Save and show
plt.savefig('ssim_comparison_exact.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved as 'ssim_comparison_exact.png'")
print("SSIM values:")
print(f"Zero-filled: {ssim_means[0]:.3f}")
print(f"MSE Loss: {ssim_means[1]:.3f} (+{mse_improvement:.3f})")
print(f"⊥-loss: {ssim_means[2]:.3f} (+{perp_improvement:.3f})")