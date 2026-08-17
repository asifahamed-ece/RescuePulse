"""
Rescue Pulse - Phase 4: int8 Post-Training Quantization for ESP32-S3 Deployment.

Loads the already-trained Keras model, converts it to a FULL int8 TFLite model
(inputs/outputs + all ops quantized, since TFLite Micro on ESP32-S3 has no float
kernel support), validates against the FULL validation set (int8 vs float32
oracle), and exports firmware artifacts (model_data.cc + model_config.h) for
on-device parity.

This script ONLY consumes existing artifacts; it does NOT retrain and does NOT
re-extract features:
  - models/siren_classifier.keras (fallback models/siren_classifier.h5)
  - datasets/features/X_train.npy, X_val.npy, y_val.npy
  - datasets/features/mfcc_stats.npz  (keys: mu, std)
"""
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

# ---------------- CONFIG ----------------
# WHY: One fixed seed makes representative-set sampling AND all numerics fully
# reproducible, so the deployed artifacts are auditable run-to-run.
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
rng = np.random.RandomState(SEED)

FEAT_DIR = Path("datasets/features")
MODEL_DIR = Path("models")

MODEL_KERAS = MODEL_DIR / "siren_classifier.keras"
MODEL_H5 = MODEL_DIR / "siren_classifier.h5"
TFLITE_PATH = MODEL_DIR / "siren_classifier_quantized.tflite"
CC_PATH = MODEL_DIR / "model_data.cc"
H_PATH = MODEL_DIR / "model_config.h"

REP_SIZE = 200          # WHY: 200 calibration samples is a good accuracy/speed tradeoff
N_WIN, N_MFCC, NUM_CLASSES = 64, 13, 2
INT8_MIN, INT8_MAX = -128, 127

# WHY: On-device parity constants (must match extract_features.py and the ESP32
# firmware). Exactly the values the task specifies for model_config.h.
RP_SR, RP_N_FFT, RP_HOP = 16000, 512, 256
RP_N_MFCC, RP_N_MELS = 13, 40
RP_FMIN, RP_FMAX, RP_N_WIN = 20, 8000, 64
RP_PRE_EMPH = 0.97

# ---------------- 1. LOAD TRAINED MODEL ----------------
# WHY: .keras is the TF>=2.13 native format; .h5 is the legacy fallback. Trying
# both keeps this script portable across the supported TF 2.10+ range.
def load_model():
    if MODEL_KERAS.exists():
        print(f"Loading model from {MODEL_KERAS}")
        return tf.keras.models.load_model(MODEL_KERAS)
    if MODEL_H5.exists():
        print(f"Loading model from {MODEL_H5}")
        return tf.keras.models.load_model(MODEL_H5)
    raise FileNotFoundError(
        "No trained model found (tried '{}' and '{}'). "
        "Run train_model.py first.".format(MODEL_KERAS, MODEL_H5)
    )

model = load_model()

# ---------------- LOAD DATA (NO EXTRACTION, NO RETRAIN) ----------------
# WHY: We only need the standardized MFCC windows + the standardization stats
# to (a) calibrate PTQ and (b) validate the int8 model against the float oracle.
X_train = np.load(FEAT_DIR / "X_train.npy")
X_val = np.load(FEAT_DIR / "X_val.npy")
y_val = np.load(FEAT_DIR / "y_val.npy")
stats = np.load(FEAT_DIR / "mfcc_stats.npz")
mu, std = stats["mu"], stats["std"]

# WHY: A corrupted/incomplete mfcc_stats.npz must fail loudly before we write
# misleading firmware headers.
assert len(mu) == N_MFCC and len(std) == N_MFCC, \
    f"mfcc_stats.npz expected length {N_MFCC}, got mu={len(mu)} std={len(std)}"
assert np.isfinite(mu).all() and np.isfinite(std).all(), "NaN/Inf in mfcc_stats.npz"

print(f"X_train {X_train.shape}  X_val {X_val.shape}  (already standardized)")

# ---------------- 2. BUILD REPRESENTATIVE DATASET ----------------
# WHY: Post-training quantization needs real data to calibrate activation
# ranges. We sample 200 training windows WITHOUT replacement (fixed seed 42)
# and feed them one-at-a-time exactly like the deployed firmware will.
n_train = int(len(X_train))
rep_size = min(REP_SIZE, n_train)
rep_idx = rng.choice(n_train, size=rep_size, replace=False)
print(f"Representative set: {rep_size} samples (seed {SEED}, no replacement)")

def representative_gen():
    """Yield one [sample.reshape(1, 64, 13)] float32 input at a time."""
    for i in rep_idx:
        sample = X_train[i]                          # (64, 13) float32, already standardized
        yield [sample.reshape(1, N_WIN, N_MFCC)]

# ---------------- 3. CONVERT (FULL int8) ----------------
# WHY: Every kernel must run as int8 on ESP32-S3 (TFLite Micro). Setting
# TFLITE_BUILTINS_INT8 + int8 inference I/O produces a graph with zero float
# tensors, and Opt.DEFAULT enables post-training quantization using the
# representative dataset for activation-range calibration.
print("\nConverting to FULL int8 TFLite...")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()
except Exception as e:
    raise RuntimeError(
        "TFLite int8 conversion FAILED. Verify the model is a valid inference "
        f"graph.\n  Reason: {e}"
    ) from e

# ---------------- 4. SAVE + SIZE COMPARISON ----------------
# WHY: The quantized bytes are the deployable artifact; the float32-vs-int8
# size delta shows the flash-memory win that makes on-device deployment viable.
TFLITE_PATH.write_bytes(tflite_bytes)

model_path_used = MODEL_KERAS if MODEL_KERAS.exists() else MODEL_H5
float_kb = os.path.getsize(model_path_used) / 1024
int8_kb = len(tflite_bytes) / 1024
print(f"\nSaved: {TFLITE_PATH}  ({int8_kb:.1f} KB)")
print(f"Size comparison: float32 {float_kb:.1f} KB  vs  int8 {int8_kb:.1f} KB "
      f"({100.0 * int8_kb / float_kb:.1f}% of float)")

# ---------------- 5. VALIDATE ON FULL VALIDATION SET ----------------
# WHY: Conversion can silently change accuracy. We compute the REAL int8
# accuracy on the entire held-out set (mimicking firmware integer math) and
# compare it against the float32 oracle to quantify the drop in percentage pts.
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
interpreter.allocate_tensors()
in_d = interpreter.get_input_details()[0]
out_d = interpreter.get_output_details()[0]

# WHY: Hard asserts guard against silently wrong device expectations - the
# deployment target REQUIRES int8 tensors with exactly these shapes.
assert in_d["dtype"] == np.int8 and tuple(in_d["shape"]) == (1, N_WIN, N_MFCC), \
    f"Unexpected input detail: dtype={in_d['dtype']} shape={in_d['shape']}"
assert out_d["dtype"] == np.int8 and tuple(out_d["shape"]) == (1, NUM_CLASSES), \
    f"Unexpected output detail: dtype={out_d['dtype']} shape={out_d['shape']}"


def _quant_params(details, name):
    """EXTRACT scale/zero_point from interpreter details.

    WHY: Newer TF/LiteRT exposes per-tensor params under
    `quantization_parameters['scales'/'zero_points']` (arrays), while older
    versions used top-level `scale`/`zero_point`. Supporting both keeps this
    script portable across TF 2.10+.
    """
    qp = details.get("quantization_parameters", {})
    if "scales" in qp and "zero_points" in qp:
        scales, zps = qp["scales"], qp["zero_points"]
        if len(scales) != 1:   # this model has single-input/single-output tensors
            raise ValueError(
                f"{name} quantization must be scalar, got {len(scales)} scales: {scales}"
            )
        return float(scales[0]), int(zps[0])
    if "scale" in details and "zero_point" in details:
        return float(details["scale"]), int(details["zero_point"])
    raise KeyError(
        f"No scale/zero_point found in {name} details; available keys: {sorted(details.keys())}"
    )


in_scale, in_zp = _quant_params(in_d, "input")
out_scale, out_zp = _quant_params(out_d, "output")
print(f"\nInterpreter I/O: in scale={in_scale} zp={in_zp} | "
      f"out scale={out_scale} zp={out_zp}")

def quantize_input(x):
    """WHY: Mirrors TFLite's input quantization (round-half-even + saturation)
    so the bytes entering the graph match what the firmware will produce."""
    q = np.round(x / in_scale) + in_zp
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)

def dequantize_output(q_out):
    """WHY: Undo the output affine map to recover float logits/softmax scores."""
    return (q_out.astype(np.int32) - out_zp).astype(np.float32) * out_scale

y_true = y_val.astype(int)
n_val = int(len(X_val))
int8_preds = np.empty(n_val, dtype=np.int64)
for i in range(n_val):
    q_in = quantize_input(X_val[i]).reshape(1, N_WIN, N_MFCC)  # (1,64,13) int8
    interpreter.set_tensor(in_d["index"], q_in)
    interpreter.invoke()
    q_out = interpreter.get_tensor(out_d["index"])              # (1,2) int8
    p = dequantize_output(q_out)
    int8_preds[i] = int(np.argmax(p))

int8_acc = float(np.mean(int8_preds == y_true))

# WHY: The float32 model is the accuracy oracle for the same validation set -
# the drop is what matters for real-world deployment decisions.
f32_probs = model.predict(X_val, verbose=0)
f32_acc = float(np.mean(np.argmax(f32_probs, axis=1) == y_true))
drop_pp = (f32_acc - int8_acc) * 100.0

print(f"\nfloat32 val accuracy: {f32_acc * 100:.2f}%")
print(f"int8   val accuracy: {int8_acc * 100:.2f}%")
print(f"accuracy drop: {drop_pp:.2f} percentage points")

# ---------------- 6. EXPORT FIRMWARE ARTIFACTS ----------------
# WHY: ESP32-S3 firmware is compiled C++; model_data.cc embeds the exact
# quantized bytes and model_config.h carries preprocessing + quantization
# parameters so on-device inference matches this validated pipeline exactly.

# 6a. models/model_data.cc  (12 hex bytes per line)
with open(CC_PATH, "w") as f:
    f.write("// Auto-generated by quantize_model.py - do not edit.\n")
    f.write("// Embedded FULL int8 TFLite model bytes for Rescue Pulse ESP32-S3.\n\n")
    f.write("unsigned char g_model_data[] = {\n")
    for i in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[i:i + 12]
        line = ", ".join(f"0x{b:02x}" for b in chunk)
        f.write("  " + line + ",\n")
    f.write("};\n")
    f.write(f"unsigned int g_model_data_len = {len(tflite_bytes)};\n")

print(f"\nExported {CC_PATH} ({len(tflite_bytes)} bytes)")

# Print the first 10 lines of model_data.cc as required.
with open(CC_PATH) as f:
    cc_lines = f.readlines()
print("First 10 lines of model_data.cc:")
for ln in cc_lines[:10]:
    print(ln.rstrip())

# 6b. models/model_config.h
# WHY: Hardcoding the audio-front-end constants guarantees the firmware's MFCC
# pipeline exactly matches the one used to produce the training features
# (train/deploy parity). The scales/zero-points are the interpreter's own
# per-tensor values used during validation.
with open(H_PATH, "w") as f:
    f.write("// Auto-generated by quantize_model.py - do not edit.\n")
    f.write("// On-device parity constants for Rescue Pulse ESP32-S3 inference.\n")
    f.write("#ifndef MODEL_CONFIG_H\n")
    f.write("#define MODEL_CONFIG_H\n\n")
    f.write("// ---------- Audio front-end (must match extract_features.py) ----------\n")
    f.write(f"#define RP_SR       {RP_SR}\n")
    f.write(f"#define RP_N_FFT    {RP_N_FFT}\n")
    f.write(f"#define RP_HOP      {RP_HOP}\n")
    f.write(f"#define RP_N_MFCC   {RP_N_MFCC}\n")
    f.write(f"#define RP_N_MELS   {RP_N_MELS}\n")
    f.write(f"#define RP_FMIN     {RP_FMIN}\n")
    f.write(f"#define RP_FMAX     {RP_FMAX}\n")
    f.write(f"#define RP_N_WIN    {RP_N_WIN}\n")
    f.write(f"#define RP_PRE_EMPH {RP_PRE_EMPH}f\n\n")
    f.write("// ---------- Per-coefficient standardization (train-set stats) ----------\n")
    f.write(f"static const float g_mfcc_mu[{N_MFCC}]  = {{" +
            ", ".join(f"{v:.6f}f" for v in mu) + "};\n")
    f.write(f"static const float g_mfcc_std[{N_MFCC}] = {{" +
            ", ".join(f"{v:.6f}f" for v in std) + "};\n\n")
    f.write("// ---------- TFLite int8 quantization parameters ----------\n")
    f.write(f"static const float g_in_scale  = {repr(in_scale)};\n")
    f.write(f"static const int   g_in_zp     = {in_zp};\n")
    f.write(f"static const float g_out_scale = {repr(out_scale)};\n")
    f.write(f"static const int   g_out_zp    = {out_zp};\n\n")
    f.write("#endif // MODEL_CONFIG_H\n")

print(f"Exported {H_PATH}")

# ---------------- 7. FINAL QUANT CHECKS ----------------
# WHY: One consolidated gate that makes it obvious whether the deployed
# artifact meets the project's hard requirements (size, accuracy, dtype).
print("\n===== QUANT CHECKS =====")
checks = [
    ("quantized size < 200 KB",          len(tflite_bytes) < 200 * 1024),
    ("accuracy drop < 2%",               (f32_acc - int8_acc) < 0.02),
    ("input tensor int8 (1,64,13)",      in_d["dtype"] == np.int8 and tuple(in_d["shape"]) == (1, N_WIN, N_MFCC)),
    ("output tensor int8 (1,2)",         out_d["dtype"] == np.int8 and tuple(out_d["shape"]) == (1, NUM_CLASSES)),
    ("model_data.cc generated",          CC_PATH.exists()),
    ("model_config.h generated",         H_PATH.exists()),
]
for name, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")