import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION (Matches your PDF Specs)
# ==========================================
ROOT = Path(__file__).resolve().parent.parent   # repo root (one level up from scripts/)
RAW_DIR = ROOT / "datasets/raw"
PROCESSED_DIR = ROOT / "datasets/processed"
MANIFEST_PATH = ROOT / "datasets/manifest.csv"

TARGET_SR = 16000       # 16kHz sample rate
TARGET_SEC = 3.0        # 3 seconds duration
TARGET_SAMPLES = int(TARGET_SR * TARGET_SEC) # 48000 samples
PEAK_NORM = 0.95        # Normalize peak amplitude to 95%
SILENCE_DB = 20         # Trim audio below -20dB

# ==========================================
# PROCESSING FUNCTION
# ==========================================
def process_audio(file_path, dest_dir, label, manifest_rows):
    try:
        # 1. Load and resample to 16kHz mono
        y, _ = librosa.load(str(file_path), sr=TARGET_SR, mono=True)
        
        if len(y) == 0:
            print(f"[SKIP] Empty file: {file_path.name}")
            return

        # 2. Trim silence from start/end
        y, _ = librosa.effects.trim(y, top_db=SILENCE_DB)
        
        # 3. Pad or Trim to exactly 3 seconds
        if len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode='constant')
        else:
            y = y[:TARGET_SAMPLES]
            
        # 4. Normalize volume (Peak normalization)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y * (PEAK_NORM / peak)
            
        # 5. Save as 16-bit PCM WAV
        clean_name = file_path.stem.replace(" ", "_").replace("-", "_") + ".wav"
        dest_path = dest_dir / clean_name
        
        # Prevent overwriting if duplicate names exist
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{clean_name[:-4]}_{counter}.wav"
            counter += 1
            
        sf.write(str(dest_path), y, TARGET_SR, subtype='PCM_16')
        
        # 6. Log to manifest
        manifest_rows.append({
            'filename': dest_path.name,
            'label': label,
            'original_file': file_path.name,
            'duration_sec': TARGET_SEC
        })
        print(f"[OK] Processed: {file_path.name} -> {dest_path.name}")
        
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path.name}: {e}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("--- Starting Audio Standardization ---")
    
    # Create output directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "siren").mkdir(exist_ok=True)
    (PROCESSED_DIR / "noise").mkdir(exist_ok=True)
    
    manifest_rows = []
    
    # Process Sirens (Label = 1)
    siren_dir = RAW_DIR / "siren"
    if siren_dir.exists():
        # rglob searches recursively through subfolders!
        files = [f for f in siren_dir.rglob("*") if f.is_file() and f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']]
        print(f"\nProcessing {len(files)} Siren files...")
        for file in files:
            process_audio(file, PROCESSED_DIR / "siren", label=1, manifest_rows=manifest_rows)
    else:
        print(f"[WARN] {siren_dir} not found!")

    # Process Noises (Label = 0)
    noise_dir = RAW_DIR / "noise"
    if noise_dir.exists():
        files = [f for f in noise_dir.rglob("*") if f.is_file() and f.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg']]
        print(f"\nProcessing {len(files)} Noise files...")
        for file in files:
            process_audio(file, PROCESSED_DIR / "noise", label=0, manifest_rows=manifest_rows)
    else:
        print(f"[WARN] {noise_dir} not found!")

    # Save Manifest & Print Summary
    if manifest_rows:
        df_manifest = pd.DataFrame(manifest_rows)
        df_manifest.to_csv(MANIFEST_PATH, index=False)
        
        total_sirens = len(df_manifest[df_manifest['label'] == 1])
        total_noises = len(df_manifest[df_manifest['label'] == 0])
        
        print("\n=========================================")
        print("STANDARDIZATION COMPLETE")
        print(f"Total Processed Sirens: {total_sirens}")
        print(f"Total Processed Noises: {total_noises}")
        print(f"All files are now: 16kHz, Mono, 3.0s, PCM_16 WAV")
        print(f"Manifest saved to: {MANIFEST_PATH}")
        print("=========================================")
    else:
        print("\n[ERROR] No files were processed!")
        print(f"Please check your folder structure at: {RAW_DIR.resolve()}")
        print("Make sure audio files (.wav, .mp3, .flac, .ogg) are actually inside the folders.")

if __name__ == "__main__":
    main()
