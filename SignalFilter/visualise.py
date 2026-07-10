import numpy as np
import pywt
import matplotlib.pyplot as plt

# --- 1. Define the Filtering Functions ---
def _remove_baseline_wander(ecg_signal, wavelet='sym8'):
    desired  = int(np.ceil(np.log2(100/0.5)))
    max_level = pywt.dwt_max_level(len(ecg_signal), pywt.Wavelet(wavelet).dec_len)
    level =  min(desired, max_level)
    
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
    coeffs_filt = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    
    baseline = pywt.waverec(coeffs_filt, wavelet)
    baseline = baseline[:len(ecg_signal)]
    
    return ecg_signal - baseline

def _denoise(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

def remove_noise_and_wander(data):
    baseline_wander = _remove_baseline_wander(data)
    cleaned = _denoise(baseline_wander)
    return cleaned


# --- 2. Load Data and Print Shapes ---
print("Loading data...")
data_train = np.load("X_train.npy")
data_valid = np.load("X_val.npy")
data_test = np.load("X_test.npy")

print("\n--- Dataset Shapes ---")
print(f"Train data shape: {data_train.shape}")
print(f"Validation data shape: {data_valid.shape}")
print(f"Test data shape: {data_test.shape}")
print("----------------------\n")


# --- 3. Plot 5 Samples Before and After ---
# We assume data_train is a 2D array of shape (num_samples, signal_length)
num_samples_to_plot = 5

# Create a figure with 5 rows and 2 columns
fig, axes = plt.subplots(nrows=num_samples_to_plot, ncols=2, figsize=(16, 12))
fig.suptitle('ECG Signal Processing: Before vs. After', fontsize=18, fontweight='bold')

for i in range(num_samples_to_plot):
    # Get the original 1D signal for the i-th sample
    original_signal = data_train[i]
    
    # Process the signal
    filtered_signal = remove_noise_and_wander(original_signal)
    
    # Plot Original (Before) - Left Column
    axes[i, 0].plot(original_signal, color='crimson', linewidth=1)
    axes[i, 0].set_title(f"Sample {i+1}: Original (Raw)")
    axes[i, 0].set_ylabel("Amplitude")
    axes[i, 0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot Filtered (After) - Right Column
    axes[i, 1].plot(filtered_signal, color='navy', linewidth=1)
    axes[i, 1].set_title(f"Sample {i+1}: Filtered (Denoised & Baseline Removed)")
    axes[i, 1].grid(True, linestyle='--', alpha=0.6)
    
    # Only show x-axis label for the bottom-most plots to keep it clean
    if i == num_samples_to_plot - 1:
        axes[i, 0].set_xlabel("Time steps")
        axes[i, 1].set_xlabel("Time steps")

# Adjust layout so titles and labels don't overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("ecg_filtering_results.png", dpi=300, bbox_inches='tight')
print("Image saved successfully as 'ecg_filtering_results.png'")

# Display the plot
plt.show()