"""Dumps EXACT librosa tables + 2 end-to-end test vectors for ESP32 parity.
Outputs into firmware/main/: mel_tables.h, test_vectors.h

NOTE: To prevent the librosa/numba vs TensorFlow LLVM segfault, TensorFlow
is strictly isolated inside subprocesses.
"""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

# ⚠️ DO NOT import tensorflow here!

SR, N_FFT, HOP, N_MFCC, N_MELS = 16000, 512, 256, 13, 40
FMIN, FMAX, PRE, N_WIN = 20, 8000, 0.97, 64
N_WIN_SAMPLES = (N_WIN - 1) * HOP + N_FFT          # 16640 = exactly one model window
PROC = Path("datasets/processed"); FEAT = Path("datasets/features")
TFL  = Path("models/siren_classifier_quantized.tflite")
OUT  = Path("firmware/main"); OUT.mkdir(parents=True, exist_ok=True)

def ortho_dct2(n_filters, width):
    """Orthonormal DCT-II matrix [n_filters][width].

    Matches scipy.fftpack.dct(norm='ortho'), which is what
    librosa.feature.mfcc() uses internally:
        D[k, n] = sqrt(1/N)                 for k=0
        D[k, n] = sqrt(2/N) * cos(pi*k*(2n+1)/(2N))  for k=1..n_filters-1
    """
    n = np.arange(width)
    k = np.arange(n_filters)[:, None]          # k = 0, 1, ..., n_filters-1
    D = np.cos(np.pi * k * (2 * n[None, :] + 1) / (2 * width))
    D[0, :] *= np.sqrt(1.0 / width)            # row 0: constant sqrt(1/N)
    D[1:, :] *= np.sqrt(2.0 / width)           # rows 1..: sqrt(2/N)
    return D

def c_float(f, name, a, cols=8):
    a = np.asarray(a, np.float32).ravel()
    f.write(f"static const float {name}[{a.size}] = {{\n")
    for i in range(0, a.size, cols):
        f.write("  " + ", ".join(f"{v:.9e}f" for v in a[i:i+cols]) + ",\n")
    f.write("};\n\n")

def c_int(f, name, a, ctype, cols=12):
    a = np.asarray(a).ravel()
    f.write(f"static const {ctype} {name}[{a.size}] = {{\n")
    for i in range(0, a.size, cols):
        f.write("  " + ", ".join(str(int(v)) for v in a[i:i+cols]) + ",\n")
    f.write("};\n\n")

# ==========================================
# TensorFlow Subprocess Snippets (Isolated)
# ==========================================
_GET_QUANT_PARAMS = r"""
import sys
import tensorflow as tf
tflite_path = sys.argv[1]
interp = tf.lite.Interpreter(model_path=tflite_path)
interp.allocate_tensors()
in_d = interp.get_input_details()[0]
qp = in_d.get("quantization_parameters", {})
if "scales" in qp and len(qp["scales"]) > 0:
    scale, zp = float(qp["scales"][0]), int(qp["zero_points"][0])
else:
    scale, zp = float(in_d.get("scale", 0.0)), int(in_d.get("zero_point", 0))
print(f"{scale} {zp}")
"""

_RUN_INFERENCE = r"""
import sys
import numpy as np
import tensorflow as tf
tflite_path, q_path = sys.argv[1], sys.argv[2]
q = np.load(q_path)
interp = tf.lite.Interpreter(model_path=tflite_path)
interp.allocate_tensors()
in_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
interp.set_tensor(in_d["index"], q)
interp.invoke()
out = interp.get_tensor(out_d["index"])
cls = int(np.argmax(out))
print(cls)
"""

def get_quant_params():
    res = subprocess.run([sys.executable, "-c", _GET_QUANT_PARAMS, str(TFL)], capture_output=True, text=True)
    if res.returncode != 0: sys.exit(f"TF Subprocess Error: {res.stderr}")
    scale, zp = res.stdout.strip().split()
    return float(scale), int(zp)

def run_tflite_inference(q_arr):
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp.name, q_arr)
        res = subprocess.run([sys.executable, "-c", _RUN_INFERENCE, str(TFL), tmp.name], capture_output=True, text=True)
        Path(tmp.name).unlink(missing_ok=True)
    if res.returncode != 0: sys.exit(f"TF Inference Error: {res.stderr}")
    return int(res.stdout.strip())

# ==========================================
# Main Execution (Librosa Safe Zone)
# ==========================================
# 1) exact tables (slaney mel + ortho DCT + periodic hamming)
with open(OUT / "mel_tables.h", "w") as f:
    f.write("#pragma once\n/* AUTO-GENERATED - DO NOT EDIT */\n\n")
    c_float(f, "g_ham", librosa.filters.get_window("hamming", N_FFT, fftbins=True))
    c_float(f, "g_mel_fb", librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX))
    c_float(f, "g_dct", ortho_dct2(N_MFCC, N_MELS))
print("[OK] mel_tables.h")

# 2) quant params + stats
in_scale, in_zp = get_quant_params()
stats = np.load(FEAT / "mfcc_stats.npz")
mu, std = stats["mu"], stats["std"]

# 3) one siren + one noise clip, exactly one window long
pick = {}
for r in csv.DictReader(open("datasets/manifest.csv", newline="", encoding="utf-8")):
    lab = int(r["label"])
    if lab in pick: continue
    p = PROC / ("siren" if lab else "noise") / r["filename"]
    if p.exists(): pick[lab] = p
    if len(pick) == 2: break

with open(OUT / "test_vectors.h", "w") as f:
    f.write("#pragma once\n#include <stdint.h>\n/* AUTO-GENERATED - DO NOT EDIT */\n\n")
    for lab, p in pick.items():
        tag = "siren" if lab else "noise"
        pcm, sr = sf.read(p, dtype="int16"); assert sr == SR
        pcm = pcm[:N_WIN_SAMPLES]
        x  = pcm.astype(np.float32) / 32768.0
        
        # Pre-emphasis (Fixed missing '*' from original script)
        xp = np.append(x[:1], x[1:] - PRE * x[:-1])
        
        M = librosa.feature.mfcc(y=xp, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP,
                                 n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
                                 window="hamming", center=False)      # (13,64) exactly
        Z = (M.T - mu) / std
        q = np.clip(np.round(Z / in_scale) + in_zp, -128, 127).astype(np.int8)
        
        # Run inference in isolated subprocess
        cls = run_tflite_inference(q.reshape(1, N_WIN, N_MFCC))
        
        f.write(f"/* {p.name} label={lab} model_says={cls} */\n#define TV_{tag.upper()}_CLASS {cls}\n")
        c_int(f, f"tv_{tag}_pcm", pcm, "int16_t")
        c_float(f, f"tv_{tag}_mfcc", M.T)     # [64][13]
        c_int(f, f"tv_{tag}_in8", q, "int8_t")
        print(f"[OK] {tag}: {p.name} -> class {cls}")

print(f"[OK] test_vectors.h (in_scale={in_scale:.6f} in_zp={in_zp})")