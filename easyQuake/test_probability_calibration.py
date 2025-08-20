#!/usr/bin/env python3
"""Test different probability calibration approaches for GPD."""

import numpy as np
import matplotlib.pyplot as plt

def temperature_scaling(logits, temperature):
    """Apply temperature scaling to logits before softmax."""
    return np.exp(logits / temperature) / np.sum(np.exp(logits / temperature), axis=-1, keepdims=True)

def apply_softmax_temperature(probs, temperature):
    """Apply temperature scaling to existing probabilities by converting back to logits."""
    # Convert probabilities back to logits (inverse softmax)
    epsilon = 1e-7  # Small value to avoid log(0)
    probs_clamped = np.clip(probs, epsilon, 1 - epsilon)
    logits = np.log(probs_clamped)
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Apply softmax
    exp_logits = np.exp(scaled_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def power_transformation(probs, power):
    """Apply power transformation to sharpen probabilities."""
    powered = np.power(probs, power)
    return powered / np.sum(powered, axis=-1, keepdims=True)

def test_probability_transformations():
    """Test different probability transformation approaches."""
    
    # Simulate current GPD probability output (similar to what we observed)
    n_samples = 1000
    
    # Create synthetic probabilities similar to our converted model
    # Current model gives roughly [0.36, 0.32, 0.32] type distributions
    np.random.seed(42)
    prob_P = np.random.uniform(0.30, 0.40, n_samples)
    prob_S = np.random.uniform(0.28, 0.38, n_samples) 
    prob_N = 1.0 - prob_P - prob_S  # Ensure they sum to 1
    
    # Stack into matrix
    current_probs = np.column_stack([prob_P, prob_S, prob_N])
    
    print("Original (converted model) probabilities:")
    print(f"P-wave: min={np.min(prob_P):.4f}, max={np.max(prob_P):.4f}, mean={np.mean(prob_P):.4f}")
    print(f"S-wave: min={np.min(prob_S):.4f}, max={np.max(prob_S):.4f}, mean={np.mean(prob_S):.4f}")
    print(f"N-wave: min={np.min(prob_N):.4f}, max={np.max(prob_N):.4f}, mean={np.mean(prob_N):.4f}")
    
    # Test different temperature values
    temperatures = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    powers = [2, 3, 4, 5, 8, 10]
    
    print("\n" + "="*60)
    print("TEMPERATURE SCALING RESULTS:")
    print("="*60)
    
    best_temp = None
    best_temp_max = 0
    
    for temp in temperatures:
        scaled_probs = apply_softmax_temperature(current_probs, temp)
        max_p = np.max(scaled_probs[:, 0])  # Max P-wave probability
        mean_p = np.mean(scaled_probs[:, 0])
        
        print(f"Temperature {temp:.1f}: P-wave max={max_p:.4f}, mean={mean_p:.4f}")
        
        if max_p > best_temp_max:
            best_temp_max = max_p
            best_temp = temp
    
    print(f"\nBest temperature: {best_temp} (max P-wave prob: {best_temp_max:.4f})")
    
    print("\n" + "="*60)
    print("POWER TRANSFORMATION RESULTS:")
    print("="*60)
    
    best_power = None
    best_power_max = 0
    
    for power in powers:
        powered_probs = power_transformation(current_probs, power)
        max_p = np.max(powered_probs[:, 0])  # Max P-wave probability
        mean_p = np.mean(powered_probs[:, 0])
        
        print(f"Power {power}: P-wave max={max_p:.4f}, mean={mean_p:.4f}")
        
        if max_p > best_power_max:
            best_power_max = max_p
            best_power = power
    
    print(f"\nBest power: {best_power} (max P-wave prob: {best_power_max:.4f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original probabilities
    axes[0,0].hist(prob_P, bins=50, alpha=0.7, color='red', label='P-wave')
    axes[0,0].hist(prob_S, bins=50, alpha=0.7, color='blue', label='S-wave')
    axes[0,0].set_title('Original (Converted Model) Probabilities')
    axes[0,0].set_xlabel('Probability')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    axes[0,0].axvline(x=0.994, color='gray', linestyle='--', label='Original threshold')
    
    # Best temperature scaling
    if best_temp:
        temp_probs = apply_softmax_temperature(current_probs, best_temp)
        axes[0,1].hist(temp_probs[:, 0], bins=50, alpha=0.7, color='red', label='P-wave')
        axes[0,1].hist(temp_probs[:, 1], bins=50, alpha=0.7, color='blue', label='S-wave')
        axes[0,1].set_title(f'Temperature Scaling (T={best_temp})')
        axes[0,1].set_xlabel('Probability')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        axes[0,1].axvline(x=0.994, color='gray', linestyle='--', label='Original threshold')
    
    # Best power transformation
    if best_power:
        power_probs = power_transformation(current_probs, best_power)
        axes[1,0].hist(power_probs[:, 0], bins=50, alpha=0.7, color='red', label='P-wave')
        axes[1,0].hist(power_probs[:, 1], bins=50, alpha=0.7, color='blue', label='S-wave')
        axes[1,0].set_title(f'Power Transformation (P={best_power})')
        axes[1,0].set_xlabel('Probability')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        axes[1,0].axvline(x=0.994, color='gray', linestyle='--', label='Original threshold')
    
    # Comparison plot
    x = np.arange(len(temperatures))
    temp_maxes = []
    for temp in temperatures:
        scaled_probs = apply_softmax_temperature(current_probs, temp)
        temp_maxes.append(np.max(scaled_probs[:, 0]))
    
    power_maxes = []
    for power in powers:
        powered_probs = power_transformation(current_probs, power)
        power_maxes.append(np.max(powered_probs[:, 0]))
    
    axes[1,1].plot(temperatures, temp_maxes, 'o-', label='Temperature scaling', color='green')
    axes[1,1].axhline(y=0.994, color='gray', linestyle='--', label='Target threshold')
    axes[1,1].set_xlabel('Temperature')
    axes[1,1].set_ylabel('Max P-wave Probability')
    axes[1,1].set_title('Temperature vs Max Probability')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/probability_calibration_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: /tmp/probability_calibration_analysis.png")
    plt.close()
    
    return best_temp, best_power

if __name__ == "__main__":
    best_temp, best_power = test_probability_transformations()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    if best_temp and best_temp < 1.0:
        print(f"✓ Temperature scaling with T={best_temp} could help restore probability calibration")
        print(f"  This would allow using a threshold closer to the original 0.994")
    
    if best_power and best_power > 1:
        print(f"✓ Power transformation with P={best_power} could also help")
        print(f"  This sharpens the probability distributions")
    
    print(f"\nAlternatively, keeping the current threshold of 0.25 is simpler and works well.")
