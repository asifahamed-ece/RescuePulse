import json, random
from pathlib import Path
import numpy as np
import librosa

# ---------------- CONFIG (must match ESP32 later) ----------------
SR, N_FFT, HOP, N_MFCC, N_MELS = 16000, 512, 256, 13, 40
FMIN, FMAX, WINDOW, PRE = 20, 8000, "hamming", 0.97
N_WIN   = 64                 # frames per model window (~1.04 s)
TRAIN_STARTS = [0, 61, 122]  # fixed windows for train clips
VAL_STARTS   = [0, 122]      # fixed windows for val clips
N_RAND_TRAIN = 2             # extra random windows per train clip
SEED = 42

MANIFEST = Path("datasets/manifest.csv")      # from Phase 1
PROCESSED = Path("datasets/processed")        # siren/ & noise/ subfolders
OUT = Path("datasets/features"); OUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED); rng = random.Random(SEED)

# ---------------- load manifest -> clip list ----------------
clips = []
for row in open(MANIFEST, encoding="utf-8"):
    pass  # placeholder; real parsing below
import csv
with open(MANIFEST, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        label = int(r["label"]); name = Path(r["filename"]).name
        cands = [PROCESSED/("siren" if label else "noise")/name,
                 PROCESSED/name, PROCESSED/r["filename"]]
        path = next((c for c in cands if c.exists()), None)
        if path: clips.append((path, label, name))
print(f"Clips found: {len(clips)}")

# ---------------- stratified clip-level split (NO leakage) ----------------
by_label = {0: [], 1: []}
for i, (p, l, n) in enumerate(clips): by_label[l].append(i)
val_idx = set()
for l, idxs in by_label.items():
    rng.shuffle(idxs)
    val_idx.update(idxs[: round(len(idxs) * 0.2)])
train_idx = [i for i in range(len(clips)) if i not in val_idx]

def mfcc_of(path):
    y, _ = librosa.load(path, sr=SR, mono=True)

    # --- Manual pre-emphasis: y[n] = x[n] - 0.97 * x[n-1] ---
    # librosa >= 0.11 removed the `preemphasis` kwarg from feature.mfcc().
    # Doing it manually keeps train/deploy parity with the ESP32 pipeline
    # (PDF Eq: y[n] = x[n] - 0.97*x[n-1]).
    y = np.append(y[:1], y[1:] - PRE * y[:-1])

    return librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        window=WINDOW, center=False
    )  # -> (13, 186)

def windows(m, starts, clip_id):
    T = m.shape[1]; out = []
    for s in starts:
        s = min(s, T - N_WIN)
        out.append((m[:, s:s+N_WIN].T, clip_id))          # (64, 13)
    return out

X_tr, y_tr, X_va, y_va, win_rows = [], [], [], [], []
for i in train_idx:
    p, l, n = clips[i]; m = mfcc_of(p)
    starts = TRAIN_STARTS + [rng.randint(0, 186-N_WIN) for _ in range(N_RAND_TRAIN)]
    for w, _ in windows(m, starts, n):
        X_tr.append(w); y_tr.append(l); win_rows.append((n, l, "train"))
for i in val_idx:
    p, l, n = clips[i]; m = mfcc_of(p)
    for w, _ in windows(m, VAL_STARTS, n):
        X_va.append(w); y_va.append(l); win_rows.append((n, l, "val"))

X_tr, y_tr = np.asarray(X_tr, np.float32), np.asarray(y_tr, np.int32)
X_va, y_va = np.asarray(X_va, np.float32), np.asarray(y_va, np.int32)

# ---------------- per-coefficient standardization (train stats only) ----------------
mu  = X_tr.reshape(-1, N_MFCC).mean(0)
std = X_tr.reshape(-1, N_MFCC).std(0) + 1e-6
X_tr = (X_tr - mu) / std
X_va = (X_va - mu) / std

# ---------------- save + verify ----------------
np.save(OUT/"X_train.npy", X_tr); np.save(OUT/"y_train.npy", y_tr)
np.save(OUT/"X_val.npy",   X_va); np.save(OUT/"y_val.npy",   y_va)
np.savez(OUT/"mfcc_stats.npz", mu=mu, std=std)
json.dump({"sr":SR,"n_fft":N_FFT,"hop":HOP,"n_mfcc":N_MFCC,"n_mels":N_MELS,
           "fmin":FMIN,"fmax":FMAX,"window":WINDOW,"n_win":N_WIN},
          open(OUT/"mfcc_config.json","w"), indent=1)

assert np.isfinite(X_tr).all() and np.isfinite(X_va).all(), "NaN/Inf in features!"
print(f"X_train {X_tr.shape}  siren%={y_tr.mean():.2f}")
print(f"X_val   {X_va.shape}  siren%={y_va.mean():.2f}")
print("Saved to", OUT)
