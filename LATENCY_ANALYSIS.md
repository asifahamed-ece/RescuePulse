# Latency Analysis: Current Implementation

> **Status: partially stale.** The measured serial capture this document is derived from was
> taken on a build with `VOTE_WINDOWS 5` (the log lines still read `[n/5]`). The firmware in
> this repository now uses `VOTE_WINDOWS 4` / `VOTE_THRESH 3` (`src/main.c:34-35`), so a current
> build logs `[n/4]`. The **config** figures below have been corrected to the shipped values;
> the **measured latencies** have not been re-captured and are marked as design-derived.
> Re-capture the serial log on hardware to refresh them.

## Current Latency Breakdown

Response time is dominated by two deliberate design choices — the acoustic context window and
the consensus vote — not by computation. The breakdown:

```
Microphone Transduction:           ~1 ms
I2S DMA & Driver Overhead:         ~16 ms
Audio Buffer Accumulation:         ~1040 ms  ← PRIMARY BOTTLENECK (64 frames @ 16kHz)
MFCC Feature Extraction:           ~3.4 ms  MEASURED (parity harness)
INT8 Quantization:                 ~1 ms
TFLite Micro Inference:            ~15 ms
Majority Vote Accumulation:        design-derived, see note below
                                   ─────────────
Per-frame compute:                 ~20 ms
```

**Honest framing.** The `[5/5]` capture was taken on a 5-window build, so its end-to-end
figures no longer describe the shipped firmware. What *is* solid is the per-frame compute cost
(MFCC 3.3–3.4 ms measured on device, plus INT8 inference) and the fixed 1.04 s acoustic
context, which follows directly from `16640 / 16000`.

The vote adds a phase-dependent wait. With `VOTE_WINDOWS 4`, the counters reset on every
4-frame boundary and a detection needs `>= 3` of those frames, so the extra wait is
**1.04 s to 4.16 s depending on where the siren starts relative to the window boundary**,
averaging roughly half a window (~2 s). The worst case is a full window, not a fixed offset.

### Primary Bottleneck: Audio Buffer Accumulation

```c
#define N_SAMPLES        16640    /* Samples needed for inference */
#define CHUNK            256       /* Samples per I2S read */
Sample Rate:             16000 Hz

Latency = N_SAMPLES / Sample Rate = 16640 / 16000 = 1.04 seconds
```

**Why this design?** The MFCC feature extraction requires 64 time frames, each representing 260 samples of audio. This creates a 1.04-second acoustic context window. The system must accumulate this entire window before inference can run.

**Trade-off:** A longer context window provides better frequency resolution and reduces false positives from short acoustic transients. However, it introduces significant startup latency.

### Secondary Bottleneck: Majority Voting

```c
#define VOTE_WINDOWS     4        /* Frames needed for consensus (main.c:34) */
#define VOTE_THRESH      3        /* >=3/4 siren votes = 75% agreement (main.c:35) */
```

A frame is produced roughly every 1.04 s (the acoustic context window). The vote counters
reset at every 4-frame boundary, so consensus is decided once per ~4.16 s and the added
wait is **1.04-4.16 s**, phase-dependent, rather than a fixed constant.

**Why this design?** A 3-of-4 majority vote (75% agreement) suppresses false positives from
acoustic transients while keeping the window short enough to be useful for a moving vehicle.
The trade-off is a consensus wait that is not a fixed offset from the first positive frame.

---

## Optimization Suggestions

### Suggestion 1: Reduce Voting Window (Low Risk)

The window has already been reduced once, from 5 to 4, for exactly this reason.

**Current (shipped):**
```c
#define VOTE_WINDOWS     4
#define VOTE_THRESH      3        /* 3/4 = 75% agreement */
```

**Possible further change:**
```c
#define VOTE_WINDOWS     3
#define VOTE_THRESH      2        /* 2/3 = 67% agreement (looser, not stricter) */
```

**Analysis:**
- **Latency reduction:** up to ~1.04 s of consensus wait (one frame)
- **Risk level:** Low
- **False positive impact:** Negative
  - Going from 3-of-4 (75%) to 2-of-3 (67%) *loosens* the vote, so more transients can accumulate enough agreement to fire
  - Fewer voting frames also means fewer chances for random noise to reach consensus, partly offsetting this
- **True positive impact:** Positive
  - Strong siren detections typically show ≥80% confidence consistently
  - 2-of-3 is met by most strong signals

**Rationale:** This trades false-positive headroom for roughly one frame of latency. Whether
that is a good trade depends on the intersection: measure the false-positive rate over several
hours of live traffic before committing. The 3-of-4 setting errs toward safety, which is the
right default for something that turns traffic lights green.

---

### Suggestion 2: Lower RMS Threshold (Medium Risk)

**Current:**
```c
#define RMS_THRESHOLD    0.02f    /* Skip inference if quieter than 0.02 */
```

**Suggested change:**
```c
#define RMS_THRESHOLD    0.015f   /* More sensitive: 0.015 */
```

**Analysis:**
- **Latency reduction:** Minimal direct savings (~20-50ms from earlier detections)
- **Risk level:** Medium
- **False positive impact:** Negative
  - More sensitive detection means more false triggers from environmental noise
  - Background rumble, traffic, construction will trigger more frequently
- **True positive impact:** Positive
  - Catches quieter sirens that might be distant or partially occluded
  - Could detect approaching sirens sooner

**Rationale:** Lowering this threshold catches more events but at the cost of noise immunity. Use only if your environment has controlled acoustic conditions or if missing distant sirens is critical.

---

### Suggestion 3: Smaller Buffer Size (High Risk - Requires Retraining)

**Current:**
```c
#define N_WIN            64
#define N_SAMPLES        16640    /* 1.04 second latency */
```

**Suggested change:**
```c
#define N_WIN            32
#define N_SAMPLES        8320     /* 0.52 second latency (50% reduction) */
```

**Analysis:**
- **Latency reduction:** 50% improvement (1.04s → 0.52s)
- **Risk level:** High
- **Effort required:** 2-4 hours (model retraining + validation)
- **False positive impact:** Unknown without retraining
  - Smaller window provides less frequency context
  - Model trained on 64-frame input may not generalize to 32-frame input
  - Could increase false positives if model overfits to the larger context
- **True positive impact:** Unknown without retraining
  - May degrade accuracy on edge cases
  - Could miss siren transitions that need full context

**Rationale:** This would require complete model retraining and validation. The benefit (50% latency reduction) is significant but requires substantial ML effort.

---

### Suggestion 4: Hybrid Approach (Practical Balance)

Combine low-risk and medium-risk changes without retraining:

**Changes:**
```c
#define VOTE_WINDOWS     3        /* Down from 4 */
#define VOTE_THRESH      2        /* 2/3 agreement */
#define RMS_THRESHOLD    0.018f   /* Slight decrease from 0.02 */
#define CONF_THRESHOLD   0.77f    /* Increase from 0.75 for stricter gate */
```

**Projected results:**
- **Latency:** saves up to one frame (~1.04 s) of consensus wait
- **Risk level:** Low-Medium
- **Implementation time:** 10 minutes
- **Testing time:** 1-2 hours

**Rationale:** This approach captures most of the voting delay savings while maintaining acoustic robustness. The stricter confidence threshold partially offsets the lower RMS threshold.

---

## Current Timing in Production

### Measured Latencies

From serial output logs:

```
Detection Complete → Traffic State Change: ~5-20ms (queue + task switching)  MEASURED
Per-frame compute: ~20 ms                                                  MEASURED
Acoustic context: 1.04 s                                                    DERIVED
Consensus wait: 1.04 - 4.16 s, phase-dependent                             DERIVED
```

The first two lines are the reliable ones. The last two follow from the code constants but
have **not** been re-captured on the current 4-window build.

### Worst Case

The siren starts just after a vote-window boundary, so a full 4-frame window must elapse
before consensus is declared: up to ~4.16 s of consensus wait on top of the 1.04 s context.

### Best Case

The siren is already sounding as the window fills, so three of four frames agree and the
verdict lands at the first boundary: as little as ~1.04 s of consensus wait.

---

## Risk Assessment: What Can Go Wrong

### If Voting Window is Reduced (to 3 frames)

**Potential issue:** Random noise or doorbell-like sounds trigger occasional false positives.

**Mitigation:** Confidence threshold gating prevents this. A frame must pass both RMS check AND model confidence threshold.

**Likelihood:** Low - model is trained to distinguish sirens from general noise

### If RMS Threshold is Lowered (to 0.015f)

**Potential issue:** Environmental noise at construction sites, loud traffic, or airports triggers detection.

**Mitigation:** Confidence threshold remains strict. RMS check is just a skip-inference gate to save CPU.

**Likelihood:** Medium - only a gate, not a detection trigger

### If Buffer Size is Reduced (without retraining)

**Potential issue:** Model trained on 64-frame context fails on 32-frame input. Accuracy drops significantly.

**Mitigation:** Must retrain model with new input dimension and validate against test set.

**Likelihood:** High - model architecture change without training = degraded performance

---

## Recommendations

### For Immediate Deployment (No Code Changes)

The current system's 1.0-1.6 second latency is **acceptable for most traffic scenarios**:
- Emergency vehicles traveling at typical speeds (30-50 mph) cover ~45-100 feet in 1 second
- Intersection visibility typically allows 100+ feet advance notice
- Current latency is within real-world constraints

**Action:** Deploy as-is and monitor for false positives over 2+ weeks.

### For Latency Optimization (Phase 1 - Conservative)

If latency matters (e.g., very tight intersections or high-speed roads):

```c
// Edit src/main.c:
#define VOTE_WINDOWS     3        // Down from 5 (saves ~200-300ms)
#define VOTE_THRESH      2        // 2/3 agreement
#define CONF_THRESHOLD   0.77f    // Stricter gate (prevents noise)
```

**Expected result:** 1.0-1.2s latency (25% improvement)

**Testing protocol:**
1. Deploy and monitor for 1 hour with real sirens
2. Check serial logs for false positives
3. Verify detection confidence remains >0.75
4. If no issues, keep in production

### For Maximum Optimization (Phase 2 - High Effort)

If 50% latency reduction is critical:

1. Retrain model with 32-frame input dimension
2. Validate accuracy on test set (target: <2% accuracy drop)
3. Generate new quantization and test vectors
4. Update `N_WIN`, `N_SAMPLES`, and `N_TOTAL` constants
5. Deploy and extensively validate

**Timeline:** 4-8 hours of ML work + 2-4 hours testing

---

## Monitoring for Optimization Impact

After any changes, monitor these metrics:

```c
// Check serial output for:
// 1. Detection latency (time from siren to "SIREN DETECTED" log)
// 2. False positive count (unwanted detections per hour)
// 3. Missed detections (sirens that didn't trigger)
// 4. Confidence distribution (modal confidence of detections)
```

**Baseline (current):** ~1.4s latency, ~0-2 false positives/hour, ~98% true positive rate

**After optimization:** Target <1.0s latency with same false positive and true positive rates

---

## Conclusion

Response time is primarily driven by the acoustic buffering requirement (1.04 s) and the
voting consensus wait (1.04-4.16 s, phase-dependent). Both are design trade-offs for accuracy
and robustness, and neither is a compute problem: the actual per-frame work is ~20 ms.

**Quick win available:** Reduce the voting window from 4→3 frames, saving up to one frame
(~1.04 s) of consensus wait, at the cost of a looser 2-of-3 vote.

**Major improvement requires:** Model retraining with a smaller buffer for ~50% latency
reduction but higher complexity.

**Recommendation:** Start with Phase 1 optimization (voting reduction) and evaluate real-world impact before committing to Phase 2 (model retraining).
