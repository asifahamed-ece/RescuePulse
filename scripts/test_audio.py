"""
Rescue Pulse - Single-file inference
Classify any audio file as SIREN or NOISE using the trained model.
Mirrors extract_features.py preprocessing exactly (pre-emphasis, MFCC,
windowing, standardization) so results match training/deploy behavior.

NOTE: librosa (-> numba/llvmlite) corrupts TensorFlow's LLVM pass registry
when both are imported in the same process (segfault). We therefore run the
librosa preprocessing in a SEPARATE subprocess and only load TF in the main
process. This keeps inference robust.

Usage:
    python test_audio.py path/to/audio.wav [--model models/siren_classifier.keras]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------- CONFIG (must match extract_features.py) ----------------
SR, N_FFT, HOP, N_MFCC, N_MELS = 16000, 512, 256, 13, 40
FMIN, FMAX, WINDOW, PRE = 20, 8000, "hamming", 0.97
N_WIN = 64                 # frames per model window (~1.04 s)
SLIDE = 32                 # window slide (overlap) for longer clips

ROOT = Path(__file__).resolve().parent.parent   # repo root (one level up from scripts/)
FEAT_DIR = ROOT / "datasets/features"
MODEL_DIR = ROOT / "models"
DEFAULT_MODEL = MODEL_DIR / "siren_classifier.keras"

# Subprocess script: computes MFCC windows + standardization, saves to .npz.
# Runs in its own process so librosa never shares a process with TF.
_PREPROCESS_SNIPPET = r"""
import sys, json
import numpy as np
import librosa

SR, N_FFT, HOP, N_MFCC, N_MELS = 16000, 512, 256, 13, 40
FMIN, FMAX, WINDOW, PRE = 20, 8000, "hamming", 0.97
N_WIN, SLIDE = 64, 32

audio_path, stats_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

# Load audio (any format, resampled to 16k mono)
y, _ = librosa.load(audio_path, sr=SR, mono=True)

# Pre-emphasis (identical to extract_features.py)
y = np.append(y[:1], y[1:] - PRE * y[:-1])

# MFCC
m = librosa.feature.mfcc(
    y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT,
    hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    window=WINDOW, center=False
)  # -> (13, T)

# Window into (N_WIN, 13) slices
T = m.shape[1]
if T < N_WIN:
    m = np.pad(m, ((0, 0), (0, N_WIN - T)), mode="edge")
    T = N_WIN
starts = list(range(0, T - N_WIN + 1, SLIDE)) or [0]
wins = [m[:, s:s + N_WIN].T for s in starts]

# Standardize with training stats
stats = np.load(stats_path)
mu, std = stats["mu"], stats["std"]
X = np.asarray(wins, np.float32)
X = (X - mu) / std

np.savez(out_path, X=X)
print(f"OK {X.shape[0]} windows")
"""


def preprocess_in_subprocess(audio_path, stats_path, out_path):
    """Run librosa preprocessing in a subprocess to avoid TF/LLVM conflict."""
    code = _PREPROCESS_SNIPPET
    result = subprocess.run(
        [sys.executable, "-c", code, str(audio_path), str(stats_path), str(out_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("[ERROR] Preprocessing subprocess failed:")
        print(result.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main():
    ap = argparse.ArgumentParser(description="Classify an audio file as SIREN/NOISE")
    ap.add_argument("audio", help="Path to audio file (wav/mp3/flac/ogg)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to .keras model")
    args = ap.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[ERROR] File not found: {audio_path}")
        sys.exit(1)

    # Preprocess in subprocess (librosa isolated from TF)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "feats.npz"
        msg = preprocess_in_subprocess(audio_path, FEAT_DIR / "mfcc_stats.npz", out_path)
        print(f"Preprocessed: {msg}")

        # Load TF + model in main process (no librosa here)
        import tensorflow as tf
        model = tf.keras.models.load_model(args.model)
        data = np.load(out_path)
        X = data["X"]

    # Predict
    probs = model.predict(X, verbose=0)          # (n_windows, 2) [noise, siren]
    preds = np.argmax(probs, axis=1)
    labels = ["NOISE", "SIREN"]

    print(f"\n{'Window':<8}{'Prediction':<10}{'Confidence':>10}")
    for i, (p, pr) in enumerate(zip(preds, probs)):
        conf = pr[p]
        print(f"{i+1:<8}{labels[p]:<10}{conf:>9.3f}")

    # Majority vote (clip-level decision)
    siren_votes = int(np.sum(preds == 1))
    total = len(preds)
    decision = 1 if siren_votes > total / 2 else 0
    mean_conf = float(np.mean(probs[:, decision]))
    print(f"\n===== CLIP DECISION: {labels[decision]} =====")
    print(f"Confidence: {mean_conf:.3f}  ({siren_votes}/{total} windows vote SIREN)")


if __name__ == "__main__":
    main()