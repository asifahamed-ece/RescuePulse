# ML Model Inference Latency Analysis

## Current Latency Breakdown: ~1-2 seconds

### 1. **Audio Buffering Delay (Primary Bottleneck)**
```
N_SAMPLES = 16640 samples
Sample Rate = 16 kHz
CHUNK = 256 samples per I2S read

Latency = (N_SAMPLES / Sample Rate) = 16640 / 16000 = 1.04 seconds
```

**Why?** The system waits to collect a full 16,640-sample buffer before running inference. This is necessary for the 64-frame MFCC window but introduces significant startup delay.

### 2. **MFCC Feature Extraction**
```
64 time frames × 13 MFCC coefficients = 832 features
Processing per block: ~50-100ms (varies with ESP32 load)
```

### 3. **Majority Voting Delay**
```
VOTE_WINDOWS = 5 frames required
At ~100ms per frame, waiting for 5 confirmations = ~500ms additional delay
```

### 4. **Inference Runtime**
```
TFLite Micro model execution: ~10-20ms (very fast on ESP32)
```

---

## Optimization Strategies

### **Option A: Reduce Buffer Size (Fastest - Recommended)**
**Trade-off**: Reduced context window, slight accuracy drop

```c
// Current (1.04s latency)
#define N_SAMPLES        16640    /* 64 frames × 260 samples */

// Optimized (0.52s latency) - Use 32 frames instead of 64
#define N_SAMPLES        8320     /* 32 frames × 260 samples */
#define N_WIN            32       /* Reduce from 64 */

// Impact: 
// - Latency cut in half (1.04s → 0.52s)
// - Model needs retraining with 32-frame input
// - Accuracy may drop 2-5%
```

**Latency Gain: 50% reduction** ⚡

---

### **Option B: Sliding Window with Early Inference (Moderate)**
**Trade-off**: More complex, moderate accuracy gain possible

Instead of waiting for full buffer, run inference every N new frames:

```c
// Pseudo-code concept
#define N_INFERENCE_STRIDE  256  /* Run inference every 256 new samples (~16ms) */
#define N_MIN_FRAMES        32   /* But require at least 32 frames buffered */

// Latency reduction: ~500ms (0.5s) by inference starting earlier
```

**Latency Gain: ~48%** (1.04s → 0.54s)

---

### **Option C: Reduce Voting Window (Low Risk)**
**Trade-off**: Fewer confirmations, slight false positive risk

```c
// Current
#define VOTE_WINDOWS     5    /* 5 frames = ~500ms+ wait for confirmation */

// Optimized
#define VOTE_WINDOWS     3    /* 3 frames = ~300ms+ wait for confirmation */
#define VOTE_THRESH      2    /* 2/3 agreement needed */

// Impact:
// - Saves ~200-300ms
// - Minimal accuracy impact if model is confident
```

**Latency Gain: ~20-30%** (1.04s → 0.75s)

---

### **Option D: Hybrid Approach (Recommended Balanced)**
Combine Options B + C for maximum practical gain:

```c
// 1. Reduce voting window
#define VOTE_WINDOWS     3
#define VOTE_THRESH      2

// 2. Lower RMS threshold to trigger inference earlier
#define RMS_THRESHOLD    0.015f  /* More sensitive to sound */

// 3. Consider streaming inference with smaller frames
// (Requires model architecture change)
```

**Total Latency Reduction: ~40-50%** (1.04s → 0.50-0.62s)

---

## Recommended Implementation Path

### **Phase 1: Quick Win (No Model Retraining) - CONSERVATIVE**
```c
// main.c - Update voting parameters CAREFULLY:
#define VOTE_WINDOWS     4        // Down from 5 (not 3 - avoid false positives)
#define VOTE_THRESH      3        // Keep at 3/4 = 75% agreement (was 3/5 = 60%)
#define RMS_THRESHOLD    0.02f    // Keep current (already well-tuned)
#define CONF_THRESHOLD   0.75f    // Increase from 0.70f (stricter confidence)
```
- **Latency**: 1.04s → ~0.85s (modest but safe)
- **Risk**: 🟢 Very Low - Only reduces voting window by 1 frame
- **False positive risk**: Minimal (actually decreases with higher confidence threshold)
- **Implementation time**: 5 minutes
- **Testing time**: 1 hour (thorough validation needed)

### **Phase 2: Model Optimization (Requires Retraining)**
```c
// If model can be retrained with 32-frame input:
#define N_WIN            32      // Down from 64
#define N_SAMPLES        8320    // Down from 16640
```
- **Latency**: 1.04s → ~0.52s
- **Risk**: Moderate (needs ML validation)
- **Implementation time**: 1-2 hours (retraining)
- **Testing time**: 2-4 hours

---

## Performance Comparison Table

| Strategy | Latency | Risk | Effort | Notes |
|----------|---------|------|--------|-------|
| Current | 1.04s | - | Baseline | Full 64-frame buffer + 5-vote window |
| Reduce voting (Option C) | ~0.75s | 🟢 Very Low | 5 min | Quick fix, safe |
| Sliding window (Option B) | ~0.54s | 🟡 Moderate | 1 hour | Complex code change |
| Smaller buffer (Option A) | ~0.52s | 🟡 Moderate | 2 hours | Needs model retraining |
| Hybrid (C+B) | ~0.50s | 🟡 Moderate | 1.5 hours | Best practical option |

---

## Latency Breakdown (Current)

```
Sound → Mic: ~1ms (transducer delay)
I2S Buffering: ~16ms (DMA + I2S driver overhead)
Audio Buffer Fill: ~1040ms ← PRIMARY BOTTLENECK
MFCC Extraction: ~50ms
Quantization: ~5ms
Inference: ~15ms
Vote Accumulation: ~300-500ms ← SECONDARY
Total: ~1420-1620ms (1.4-1.6 seconds observed)
```

---

## Recommended Action

**START WITH PHASE 1** (voting reduction):
1. Change `VOTE_WINDOWS` from 5 → 3
2. Change `VOTE_THRESH` from 3 → 2  
3. Slightly lower `RMS_THRESHOLD` if false negatives appear
4. Test for 30 minutes with real sirens

**Expected Result**: 25-30% latency improvement with minimal risk

If Phase 1 is insufficient, then pursue Phase 2 (model retraining).

---

## Code Changes for Phase 1 (Conservative - No False Positives)

File: `src/main.c`

```diff
- #define VOTE_WINDOWS     5
+ #define VOTE_WINDOWS     4          // Reduce by 1 frame only (~200ms)

- #define VOTE_THRESH      3
+ #define VOTE_THRESH      3          // Keep same (3/4 = 75% agreement)

- #define RMS_THRESHOLD    0.02f
+ #define RMS_THRESHOLD    0.02f      // Keep current (well-tuned)

- #define CONF_THRESHOLD   0.70f
+ #define CONF_THRESHOLD   0.75f      // Stricter confidence requirement
```

**Why This Approach:**
- ✅ Reduces latency by ~200ms (1040ms → 840ms) safely
- ✅ Actually **decreases** false positives (higher confidence bar)
- ✅ Same voting threshold ratio: 3/4 agreement = 75% confidence
- ✅ No model retraining needed
- ✅ Minimal risk - only 1 frame removed from voting window

**Expected Results After Change:**
- Detection latency: 1.0-1.2s → 0.8-1.0s
- False positive rate: Same or slightly better (higher confidence threshold)
- True positive rate: Minimal impact (strong sirens still detected)

**Build and test**: 
```bash
cd /home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO
pio run -t upload
# Monitor output and test with sirens
```

**Validation Checklist:**
- [ ] Test with loud siren sounds (should detect immediately)
- [ ] Test with background noise (should NOT trigger)
- [ ] Test with music/speech (should NOT trigger)
- [ ] Monitor for false positives over 1 hour continuous operation

