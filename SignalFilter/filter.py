import numpy as np
import pywt
from scipy.signal import butter, filtfilt, iirnotch, medfilt, find_peaks
from datasets import Dataset, DatasetDict

# 1. Load the Data
print("Loading datasets...")
data_train = np.load("X_train.npy")
data_valid = np.load("X_val.npy")
data_test = np.load("X_test.npy")

# 2. Define the Processing Functions
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
    # Wavelet transform
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # Threshold denoising
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # Inverse wavelet transform to obtain the denoised signal
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

def remove_noise_and_wander(data):
    baseline_wander = _remove_baseline_wander(data)
    cleaned = _denoise(baseline_wander)
    return cleaned

# 3. Apply the Filter
# Assuming your data shape is (num_samples, signal_length). 
# If it's a 1D array of a single continuous signal, remove the list comprehension loop.
def process_batch(dataset_array):
    print(f"Processing {len(dataset_array)} samples...")
    # Applies the cleaning function to each row/signal in the array
    return [remove_noise_and_wander(signal) for signal in dataset_array]

print("\nFiltering Train Data...")
train_filtered = process_batch(data_train)

print("Filtering Validation Data...")
valid_filtered = process_batch(data_valid)

print("Filtering Test Data...")
test_filtered = process_batch(data_test)

# 4. Package for Hugging Face
print("\nPackaging datasets...")
train_ds = Dataset.from_dict({"ecg_signal": train_filtered})
valid_ds = Dataset.from_dict({"ecg_signal": valid_filtered})
test_ds = Dataset.from_dict({"ecg_signal": test_filtered})

hf_dataset = DatasetDict({
    "train": train_ds,
    "validation": valid_ds,
    "test": test_ds
})

# 5. Push to Hub
# Replace 'your-username/ecg-filtered-dataset' with your actual HF username and desired repo name
REPO_ID = "Codemaster67/ECGFiltered"

print(f"\nPushing to Hugging Face Hub at: {REPO_ID}...")
hf_dataset.push_to_hub(REPO_ID)
print("Push complete! Data is now live on Hugging Face.")