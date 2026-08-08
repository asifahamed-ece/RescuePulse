"""
Rescue Pulse - Phase 3: Model Architecture & Training
Upgraded Conv1D CNN for (64, 13) MFCC windows.
Target: val acc >90%, precision >90%, recall >85%, F1 >87%, int8 <200KB.
v2: L2 regularization, stronger dropout, lower LR, clip-level majority-vote eval.
"""
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe plotting (no GUI needed)
import matplotlib.pyplot as plt
import tensorflow as tf

# ---------------- CONFIG ----------------
SEED = 42
tf.keras.utils.set_random_seed(SEED)

FEAT_DIR = Path("datasets/features")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE, EPOCHS = 32, 50
L2_REG = 1e-4
DROPOUT_1 = 0.4
DROPOUT_2 = 0.2
LR = 5e-4

# ---------------- LOAD DATA ----------------
X_train = np.load(FEAT_DIR / "X_train.npy")
y_train = np.load(FEAT_DIR / "y_train.npy")
X_val   = np.load(FEAT_DIR / "X_val.npy")
y_val   = np.load(FEAT_DIR / "y_val.npy")
clip_tr = np.load(FEAT_DIR / "clip_tr.npy")
clip_va = np.load(FEAT_DIR / "clip_va.npy")
print(f"X_train {X_train.shape}  X_val {X_val.shape}")

# ---------------- BUILD MODEL ----------------
# WHY: Conv1D over time axis = cheap temporal pattern detector.
# GlobalAvgPool keeps params tiny; BN stabilizes training; Dropout fights overfit.
# v2: L2 weight decay on all conv/dense, higher dropout, smaller hidden layer.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 13)),
    tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),                      # -> (32, 64)
    tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),                      # -> (16, 128)
    tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(DROPOUT_2),                   # v2: extra dropout pre-pool
    tf.keras.layers.GlobalAveragePooling1D(),             # -> (128,)
    tf.keras.layers.Dense(64, activation="relu",          # v2: 128 -> 64
                          kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
    tf.keras.layers.Dropout(DROPOUT_1),                   # v2: 0.3 -> 0.4
    tf.keras.layers.Dense(2, activation="softmax"),       # [noise, siren]
])

model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ---------------- TRAIN ----------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=1),
]
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=BATCH_SIZE, epochs=EPOCHS,
                    callbacks=callbacks)

# PDF safety rule: flag data issue if val acc < 85% after 20 epochs
if np.max(history.history["val_accuracy"][:20]) < 0.85:
    print("[WARN] val accuracy < 85% after 20 epochs -> likely DATA issue, not model issue.")

# ---------------- EVALUATE (window-level) ----------------
y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
y_true = y_val.astype(int)
acc = float(np.mean(y_pred == y_true))

def report(y_true, y_pred):
    """Dependency-free precision/recall/F1 per class."""
    print(f"\n{'class':<7}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}")
    out = {}
    for lab, name in [(0, "noise"), (1, "siren")]:
        tp = int(np.sum((y_pred == lab) & (y_true == lab)))
        fp = int(np.sum((y_pred == lab) & (y_true != lab)))
        fn = int(np.sum((y_pred != lab) & (y_true == lab)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[name] = (p, r, f)
        print(f"{name:<7}{p:7.3f}{r:7.3f}{f:7.3f}{tp + fn:9d}")
    return out

rep = report(y_true, y_pred)
cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=2).numpy()
print("\nConfusion Matrix [rows=true, cols=pred]:\n", cm)

# ---------------- EVALUATE (clip-level majority vote) ----------------
# Real-world classification = one decision per clip, not per 1s window.
# Majority vote over the 5 val windows per clip.
clip_ids = np.unique(clip_va)
clip_true, clip_pred = [], []
for cid in clip_ids:
    mask = clip_va == cid
    votes = y_pred[mask]
    clip_true.append(int(y_val[mask][0]))          # all windows share clip label
    clip_pred.append(int(np.bincount(votes).argmax()))
clip_true, clip_pred = np.array(clip_true), np.array(clip_pred)
clip_acc = float(np.mean(clip_true == clip_pred))
print(f"\n===== CLIP-LEVEL (majority vote over {len(clip_ids)} val clips) =====")
print(f"Clip accuracy: {clip_acc:.3f}")
rep_clip = report(clip_true, clip_pred)

# ---------------- PLOTS ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(history.history["accuracy"], label="train")
a1.plot(history.history["val_accuracy"], label="val")
a1.set_title("Accuracy"); a1.legend(); a1.grid(True)
a2.plot(history.history["loss"], label="train")
a2.plot(history.history["val_loss"], label="val")
a2.set_title("Loss"); a2.legend(); a2.grid(True)
fig.savefig(MODEL_DIR / "training_curves.png", dpi=150)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["noise", "siren"]); ax.set_yticks([0, 1], ["noise", "siren"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)
ax.set_ylabel("True"); ax.set_xlabel("Predicted"); ax.set_title("Confusion Matrix")
fig.colorbar(im); fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)

# ---------------- SAVE ----------------
major, minor = map(int, tf.__version__.split(".")[:2])
ext = ".keras" if (major, minor) >= (2, 13) else ".h5"   # TF>=2.13 prefers .keras
model_path = MODEL_DIR / f"siren_classifier{ext}"
model.save(model_path)

params = model.count_params()
print(f"\nSaved: {model_path} ({os.path.getsize(model_path)/1024:.0f} KB float32)")
print(f"Params: {params:,}  |  est. int8 size: ~{params/1024:.0f} KB")

# ---------------- TARGET CHECKS ----------------
gap = history.history["accuracy"][-1] - history.history["val_accuracy"][-1]
checks = [
    ("val accuracy > 90%",      acc > 0.90),
    ("siren precision > 90%",   rep["siren"][0] > 0.90),
    ("siren recall > 85%",      rep["siren"][1] > 0.85),
    ("siren F1 > 87%",          rep["siren"][2] > 0.87),
    ("train-val gap < 5%",      gap < 0.05),
    ("int8 estimate < 200KB",   params / 1024 < 200),
    ("clip-level acc > 90%",    clip_acc > 0.90),
]
print("\n===== TARGET CHECKS =====")
for name, ok in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")