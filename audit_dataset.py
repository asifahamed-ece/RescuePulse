"""
Rescue Pulse - Data Audit & Dedup
Detects label contamination: identical audio content labeled BOTH siren AND noise.
Resolves by dropping ALL rows in a conflicting group (we cannot trust either label).
Writes a cleaned manifest to datasets/manifest_clean.csv.
"""
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

MANIFEST = Path("datasets/manifest.csv")
PROCESSED = Path("datasets/processed")
OUT = Path("datasets/manifest_clean.csv")


def file_hash(path: Path, chunk=1 << 20) -> str:
    """SHA-256 of file bytes (content-based, not name-based)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
    print(f"Manifest rows: {len(rows)}")

    # ---- Pass 1: same-name collisions with conflicting labels ----
    by_name = defaultdict(list)
    for r in rows:
        by_name[r["filename"]].append(r)

    name_conflicts = []
    for name, group in by_name.items():
        labels = {r["label"] for r in group}
        if len(labels) > 1:
            name_conflicts.append((name, group))
    print(f"Name collisions with conflicting labels: {len(name_conflicts)}")

    # ---- Pass 2: FULL content-hash dedup across ALL files (any name) ----
    # Hash every processed file once, then find any hash seen under BOTH labels.
    print("Hashing all processed files...")
    hash_to_rows = defaultdict(list)   # hash -> [(label, filename)]
    for r in rows:
        label = r["label"]
        cand = PROCESSED / ("siren" if label == "1" else "noise") / r["filename"]
        if cand.exists():
            h = file_hash(cand)
            hash_to_rows[h].append((label, r["filename"]))

    # True conflict = same audio content under different labels (any filename)
    true_conflicts = []
    for h, items in hash_to_rows.items():
        labels = {lab for lab, _ in items}
        if len(labels) > 1:
            true_conflicts.append((h, items))

    print(f"TRUE content duplicates with conflicting labels: {len(true_conflicts)}")

    # Report
    for h, items in true_conflicts:
        print(f"  [CONFLICT] hash={h[:12]}  rows={items}")

    # Resolve: drop ALL rows whose content hash is in a true-conflict group
    drop_hashes = {h for h, _ in true_conflicts}
    kept = [r for r in rows if not (
        (PROCESSED / ("siren" if r["label"] == "1" else "noise") / r["filename"]).exists()
        and file_hash(PROCESSED / ("siren" if r["label"] == "1" else "noise") / r["filename"]) in drop_hashes
    )]

    # Also drop any rows whose processed file is missing (safety)
    missing = []
    final = []
    for r in kept:
        label = r["label"]
        cand = PROCESSED / ("siren" if label == "1" else "noise") / r["filename"]
        if cand.exists():
            final.append(r)
        else:
            missing.append(r["filename"])

    # Write cleaned manifest
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(final)

    # Summary
    from collections import Counter
    c_old = Counter(r["label"] for r in rows)
    c_new = Counter(r["label"] for r in final)
    print("\n===== AUDIT SUMMARY =====")
    print(f"Before: siren={c_old['1']}  noise={c_old['0']}  total={len(rows)}")
    print(f"After:  siren={c_new['1']}  noise={c_new['0']}  total={len(final)}")
    dropped = len(rows) - len(final)
    print(f"Dropped conflicts: {dropped} rows "
          f"({len(true_conflicts)} content-conflict groups, {len(missing)} missing files)")
    print(f"Cleaned manifest -> {OUT}")


if __name__ == "__main__":
    main()