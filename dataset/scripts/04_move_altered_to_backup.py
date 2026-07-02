"""
Move AI-altered PIL images to backup folder.
Keeps them for reference but removes from the active dataset path.
"""
import shutil, time
from pathlib import Path

src = Path("dataset/images/ai_altered")
dst = Path("dataset/backup_pil_alterations")

if dst.exists():
    print(f"Backup already exists at {dst}")
    print("Remove it first if you want to re-run this script")
    exit(1)

print(f"Moving {src} to {dst} ...")
t0 = time.time()

dst.parent.mkdir(exist_ok=True)
shutil.move(str(src), str(dst))

elapsed = time.time() - t0
print(f"Done in {elapsed:.0f}s")
print(f"Moved to: {dst.resolve()}")
