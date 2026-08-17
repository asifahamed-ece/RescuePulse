import json, random
from pathlib import Path
import numpy as np
import librosa

# ---------------- CONFIG (must match ESP32 later) ----------------
SR, N_FFT, HOP, N_MFCC, N_MELS = 16000, 512, 256, 13, 40
FMIN, FMAX, WINDOW, PRE = 20, 8000, "hamming", 0.97
N_WIN   = 64                 # frames per model window (~1.04 s)
TRAIN_STARTS = [0, 61, 122]  # fixed windows for train clips
VAL_STARTS   = [0, 31, 62, 93, 124]  # 5 spread windows -> clip-level majority vote
N_RAND_TRAIN = 3             # extra random windows per train clip
N_AUG_TRAIN  = 4             # augmented copies per train window
AUG_SHIFT_MAX = 15           # max random time-shift in frames
AUG_MASK_T   = 2             # max consecutive time frames to zero (SpecAugment)
AUG_MASK_F   = 1             # max coeff frames to zero  (SpecAugment)
AUG_JITTER   = 0.05          # gaussian noise sigma on standardized features
AUG_PROB     = 0.9           # probability a window gets augmented
SEED = 42

ROOT = Path(__file__).resolve().parent.parent      # repo root (one level up from scripts/)
MANIFEST = ROOT / "datasets/manifest_clean.csv"    # audit output (falls back below)
MANIFEST_FALLBACK = ROOT / "datasets/manifest.csv"
PROCESSED = ROOT / "datasets/processed"            # siren/ & noise/ subfolders
OUT = ROOT / "datasets/features"; OUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED); rng = random.Random(SEED)

# ---------------- load manifest -> clip list ----------------
manifest = MANIFEST if MANIFEST.exists() else MANIFEST_FALLBACK
print(f"Using manifest: {manifest}")

import csv
clips = []
with open(manifest, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        label = int(r["label"]); name = Path(r["filename"]).name
        cands = [PROCESSED/("siren" if label else "noise")/name,
                 PROCESSED/name, PROCESSED/r["filename"]]
        path = next((c for c in cands if c.exists()), None)
        # Unique clip id: label+name (same filename exists under both labels)
        if path: clips.append((path, label, f"{label}_{name}"))
print(f"Clips found: {len(clips)}")

# ---------------- stratified clip-level split (NO leakage) ----------------
by_label = {0: [], 1: []}
for i, (p, l, n) in enumerate(clips): by_label[l].append(i)
val_idx = set()
for l, idxs in by_label.items():
    rng.shuffle(idxs)
    val_idx.update(idxs[: round(len(idxs) * 0.2)])
train_idx = [i for i in range(len(clips)) if i not in val_idx]
print(f"Train clips: {len(train_idx)}  Val clips: {len(val_idx)}")

def mfcc_of(path):
    y, _ = librosa.load(path, sr=SR, mono=True)

    # --- Manual pre-emphasis: y[n] = x[n] - 0.97 * x[n-1] ---
    # librosa >= 0.11 removed the `preemphasis` kwarg from feature.mfcc().
    # Doing it manually keeps train/deploy parity with the ESP32 pipeline.
    y = np.append(y[:1], y[1:] - PRE * y[:-1])

    return librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        window=WINDOW, center=False
    )  # -> (13, 186)

def windows(m, starts):
    T = m.shape[1]; out = []
    for s in starts:
        s = min(s, T - N_WIN)
        out.append(m[:, s:s+N_WIN].T)          # (64, 13)
    return out

def augment(w):
    """Feature-space augmentation of a (64,13) window. Returns a copy."""
    a = w.copy()
    # 1) Random time-shift (roll frames maintaining temporal structure)
    shift = rng.randint(-AUG_SHIFT_MAX, AUG_SHIFT_MAX)
    a = np.roll(a, shift, axis=0)
    # 2) SpecAugment: zero random time band
    if AUG_MASK_T > 0:
        t0 = rng.randint(0, N_WIN - AUG_MASK_T)
        a[t0:t0 + AUG_MASK_T, :] = 0.0
    # 3) SpecAugment: zero random freq band
    if AUG_MASK_F > 0:
        f0 = rng.randint(0, N_MFCC - AUG_MASK_F)
        a[:, f0:f0 + AUG_MASK_F] = 0.0
    # 4) Small gaussian jitter
    if AUG_JITTER > 0:
        a += rng.gauss(0.0, AUG_JITTER) * np.ones_like(a)
    return a

# ---------------- build feature arrays ----------------
X_tr, y_tr, X_va, y_va = [], [], [], []
clip_ids_tr, clip_ids_va = [], []       # per-window clip id for majority voting

for i in train_idx:
    p, l, n = clips[i]
    m = mfcc_of(p)
    starts = TRAIN_STARTS + [rng.randint(0, 186 - N_WIN) for _ in range(N_RAND_TRAIN)]
    for w in windows(m, starts):
        X_tr.append(w); y_tr.append(l); clip_ids_tr.append(n)
        # augmented copies
        for _ in range(N_AUG_TRAIN):
            if rng.random() < AUG_PROB:
                X_tr.append(augment(w)); y_tr.append(l); clip_ids_tr.append(n)
for i in val_idx:
    p, l, n = clips[i]
    m = mfcc_of(p)
    for w in windows(m, VAL_STARTS):
        X_va.append(w); y_va.append(l); clip_ids_va.append(n)

X_tr, y_tr = np.asarray(X_tr, np.float32), np.asarray(y_tr, np.int32)
X_va, y_va = np.asarray(X_va, np.float32), np.asarray(y_va, np.int32)

# ---------------- per-coefficient standardization (train stats only) ----------------
mu  = X_tr.reshape(-1, N_MFCC).mean(0)
std = X_tr.reshape(-1, N_MFCC).std(0) + 1e-6
X_tr = (X_tr - mu) / std
X_va = (X_va - mu) / std
# clip_ids -> int codes for compact storage
id_map = {n: i for i, n in enumerate(sorted(set(clip_ids_tr) | set(clip_ids_va)))}
c_tr = np.array([id_map[n] for n in clip_ids_tr], np.int32)
c_va = np.array([id_map[n] for n in clip_ids_va], np.int32)

# ---------------- save + verify ----------------
np.save(OUT/"X_train.npy", X_tr); np.save(OUT/"y_train.npy", y_tr)
np.save(OUT/"X_val.npy",   X_va); np.save(OUT/"y_val.npy",   y_va)
np.save(OUT/"clip_tr.npy", c_tr); np.save(OUT/"clip_va.npy", c_va)
np.savez(OUT/"mfcc_stats.npz", mu=mu, std=std)
json.dump({"sr":SR,"n_fft":N_FFT,"hop":HOP,"n_mfcc":N_MFCC,"n_mels":N_MELS,
           "fmin":FMIN,"fmax":FMAX,"window":WINDOW,"n_win":N_WIN},
          open(OUT/"mfcc_config.json","w"), indent=1)

assert np.isfinite(X_tr).all() and np.isfinite(X_va).all(), "NaN/Inf in features!"
print(f"X_train {X_tr.shape}  siren%={y_tr.mean():.2f}  (augmented x{N_AUG_TRAIN+1})")
print(f"X_val   {X_va.shape}  siren%={y_va.mean():.2f}  (5 windows/clip)")
print(f"Clips: train={len(set(clip_ids_tr))}  val={len(set(clip_ids_va))}")
print("Saved to", OUT)