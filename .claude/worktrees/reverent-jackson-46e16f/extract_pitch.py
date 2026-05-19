import torch
import torchcrepe
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# -----------------------------
# 1. Load Audio
# -----------------------------
file_path = "Balagopala - Bhairavi - Dikshitar.wav"   # your input file
audio, sr = librosa.load(file_path, sr=16000, mono=True)  # CREPE prefers 16kHz mono
audio = torch.tensor(audio).unsqueeze(0)  # shape (1, n_samples)

# -----------------------------
# 2. Pitch Extraction with torchcrepe
# -----------------------------
hop_length = 160   # 10 ms hop (160 samples at 16kHz)
fmin, fmax = 50.0, 1000.0   # typical Carnatic vocal range

with torch.no_grad():
    f0, pd = torchcrepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model='full',      # 'tiny' is faster but less accurate
        batch_size=128,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        return_periodicity=True
    )

f0 = f0.squeeze().cpu().numpy()          # shape (n_frames,)
pd = pd.squeeze().cpu().numpy()

# Mask out unvoiced frames
f0[pd < 0.9] = np.nan

# -----------------------------
# 3. Tonic Estimation
# -----------------------------
# Build pitch histogram in cents (log scale)
valid_f0 = f0[~np.isnan(f0)]
cents = 1200 * np.log2(valid_f0 / 55.0)  # relative to 55 Hz A
hist, bin_edges = np.histogram(cents, bins=300)

# Pick the highest peak as tonic
tonic_bin = bin_edges[np.argmax(hist)]
tonic_hz = 55.0 * 2**(tonic_bin / 1200.0)
print("Estimated tonic (Sa):", tonic_hz, "Hz")

# -----------------------------
# 4. Tonic Normalization
# -----------------------------
norm_cents = 1200 * np.log2(f0 / tonic_hz)
# Wrap to within [-600, 1200] for ~2 octaves around tonic
norm_cents = np.mod(norm_cents + 1200, 2400) - 1200

# -----------------------------
# 5. Contour Smoothing
# -----------------------------
# Median filter to remove jitter
smooth_cents = median_filter(norm_cents, size=5)

# Optional: Piecewise Aggregate Approximation (PAA) per half-matra
# Example: downsample to 20 Hz
downsample_factor = int(sr / hop_length / 20)
paa_cents = smooth_cents[::downsample_factor]

# -----------------------------
# 6. Visualization
# -----------------------------
times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

plt.figure(figsize=(12,4))
plt.plot(times, norm_cents, label="Raw tonic-normalized", alpha=0.5)
plt.plot(times, smooth_cents, label="Smoothed", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Pitch (cents relative to Sa)")
plt.legend()
plt.title("Pitch Extraction for Carnatic Audio")
plt.show()
