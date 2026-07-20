"""Download all RusLawOD parquet files for analysis."""
import os, sys, time
from huggingface_hub import hf_hub_download

REPO = "irlspbru/RusLawOD"
FILES = [f"ruslawod_{i:02d}.parquet" for i in range(1, 12)]
DEST = "/home/vllm/rag/raw"

os.makedirs(DEST, exist_ok=True)

for f in FILES:
    dst = os.path.join(DEST, f)
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print(f"[SKIP] {f} already exists ({os.path.getsize(dst)//1024//1024} MB)")
        continue
    t0 = time.time()
    print(f"[DOWNLOAD] {f}...", end=" ", flush=True)
    path = hf_hub_download(REPO, f, repo_type="dataset", local_dir=DEST, local_dir_use_symlinks=False)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(path) / 1024 / 1024
    speed = size_mb / elapsed if elapsed > 0 else 0
    print(f"done ({size_mb:.0f} MB in {elapsed:.0f}s, {speed:.1f} MB/s)")

print(f"\nAll files in {DEST}:")
os.system(f"ls -lh {DEST}/ruslawod_*.parquet | awk '{{print $5, $9}}'")
