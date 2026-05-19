import torch
import torchcrepe
import librosa
import numpy as np
import matplotlib.pyplot as plt


print("Hello")
# 1. Load audio
audio, sr = librosa.load( r"Balagopala - Bhairavi - Dikshitar.wav",sr=16000)
print("Loaded audio with sample rate:", sr)
# Convert to torch tensor with batch dimension (float32)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

# 2. Extract pitch
hop_length = int(sr / 50)  # 10ms hop (100 Hz frame rate)
fmin, fmax = 50, 550

frequency, confidence = torchcrepe.predict(
    audio,
    sr,
    hop_length,
    fmin=fmin,
    fmax=fmax,
    model='full',
    batch_size=128,
    device=device,
    return_periodicity=True,
)

print("Pitch extraction complete.")

# Convert to numpy
frequency = frequency.cpu().numpy().squeeze()
confidence = confidence.cpu().numpy().squeeze()

# Compute time axis
frames = frequency.shape[-1]
time = np.arange(frames) * hop_length / sr

# --- 3. Filter out low-confidence pitches ---
# --- 3. Filter out low-confidence pitches ---
confidence_threshold = 0.8
frequency[confidence < confidence_threshold] = np.nan

# --- 3a. Interpolate missing values for smooth curve ---
valid_idx = ~np.isnan(frequency)
frequency_interp = np.interp(
    np.arange(len(frequency)),
    np.arange(len(frequency))[valid_idx],
    frequency[valid_idx]
)

# --- 4. Visualize and Save the Pitch Contour ---
print("Generating plot...")
plt.figure(figsize=(12, 6))

# Smoothed line (interpolated)
plt.plot(time, frequency_interp, color='blue', alpha=0.7, linewidth=2,
         label='Smoothed Contour')

# Raw points (with gaps)
plt.scatter(time, frequency, s=5, color='red', alpha=0.5,
            label='Raw CREPE Output')

plt.title('Fundamental Frequency (F0) Contour')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.ylim(fmin, fmax)

# Save plot
output_image_path = 'pitch_contour.png'
plt.savefig(output_image_path, dpi=300)
print(f"Plot saved to {output_image_path}")

# Display plot
plt.show()
