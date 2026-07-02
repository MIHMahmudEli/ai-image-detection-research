"""
Download GenImage by generator with progress tracking, resume, and extraction.
Usage: python scripts/06_download_genimage.py --generator BigGAN
"""
import argparse, sys, time, json, shutil, io, struct
from pathlib import Path
from zipfile import ZipFile

GENIMAGE_OUT = Path("dataset/images/genimage_ai")
GENIMAGE_OUT.mkdir(parents=True, exist_ok=True)

GENERATORS = {
    "BigGAN": [
        ("1dtuXpn2MDGRVSrzZwZvKWtQA6yO3U6u5", "imagenet_ai_0419_biggan.z01"),
        ("1-ECWWWXDEwDlBz4CBALSThZfyg0eGBm9", "imagenet_ai_0419_biggan.z02"),
        ("16PPUE9XcHndSprOcTPHVuDsZPyApyfF_", "imagenet_ai_0419_biggan.z03"),
        ("1pOyRWuy5U7aHhELEuu_0dzbxdq4ZxWYR", "imagenet_ai_0419_biggan.z04"),
        ("1mJDGWC4yYqViT_4V2v56Ww8G39G7LoF0", "imagenet_ai_0419_biggan.z05"),
        ("1d32eMbi9T2McdSPDfMZWk25NNrIvmK90", "imagenet_ai_0419_biggan.z06"),
        ("1dAy2qpQmh7c7Ye6Awje-Wku7LaJUxFpS", "imagenet_ai_0419_biggan.z07"),
        ("1TRTjKdrJPEvLaP-O_31nrlnAsqRO6n8R", "imagenet_ai_0419_biggan.zip"),
    ],
    "VQDM": [
        ("12hRuzHQ-eAbrJeGS01n1iYNHYYl0A0LV", "imagenet_ai_0419_vqdm.z01"),
        ("1y_fjkvzTWHXAbzIurRmdQ_rU54Dj0B45", "imagenet_ai_0419_vqdm.z02"),
        ("15pVNrnfoFebBjM4utOW1UuzyKVBsUxjC", "imagenet_ai_0419_vqdm.z03"),
        ("1xKpZOC9Rbe0EQMH6HxhmqfZeI0y13Opq", "imagenet_ai_0419_vqdm.z04"),
        ("1Yjsi-auGqvOZV3r6hfcrgoHbWVLag6jN", "imagenet_ai_0419_vqdm.z05"),
        ("1tm0ierEBLH18ZqviTy8Ds86Fu1lTrIrr", "imagenet_ai_0419_vqdm.z06"),
        ("1ZT_l-jRj-1GyXnuaBxGCZKzjk0TCZDqV", "imagenet_ai_0419_vqdm.z07"),
        ("1Y0B6THLU6ZZPcXx_Jc62eZ2h9-9qw96h", "imagenet_ai_0419_vqdm.z08"),
        ("18WPZg86aw7fKdA7LU-lCztiDrggAeF4s", "imagenet_ai_0419_vqdm.z09"),
        ("19R3BOE_XpthkH3ia8CXE2SVsJZHjfMA_", "imagenet_ai_0419_vqdm.z10"),
        ("1WeH5_1ysaqmDvgwoX1yUmTSLj_BkTZg-", "imagenet_ai_0419_vqdm.z11"),
        ("1T4rGc6S3cb5m4PrmJKkvJZnVd92wLYWh", "imagenet_ai_0419_vqdm.zip"),
    ],
    "glide": [
        ("1NvEr9FSMhc28qxt2RsfSHyRoVm06YbO6", "imagenet_glide.z01"),
        ("15BwJz-Iq5ealHgtiFau10n28WhRGvGGR", "imagenet_glide.z02"),
        ("1O0uEfzUlYcojOVqTpjF7AI3P9mFU2ooE", "imagenet_glide.z03"),
        ("1SzGYBOYsWX3SwwJt51oZIDIzt3JBzzVA", "imagenet_glide.z04"),
        ("1Du_7BEFuXzKGqI-bSvi3FWXWlFmUf5K3", "imagenet_glide.z05"),
        ("1TVTNRKtS3Tl1ovmej5u9Flg7yqHgCadQ", "imagenet_glide.z06"),
        ("1LssUx4DjLOHrv8hD8MCY4u1ADfHrOXHc", "imagenet_glide.z07"),
        ("1ws3dqHTCAwteaf950YWMGl_MYKr0kTp3", "imagenet_glide.z08"),
        ("1iUtbABfaGZPGxgJDmFFqgsYqogUFFtma", "imagenet_glide.z09"),
        ("1IAo3GYGFR3vbExTjCZxwQ9RPdYhoNncA", "imagenet_glide.z10"),
        ("1npimXkkd74IkzkBChduvlkQWQTUhP8CA", "imagenet_glide.zip"),
    ],
}

def fmt_bytes(b):
    for u in ["B", "KB", "MB", "GB"]:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"


class SplitZipReader(io.RawIOBase):
    def __init__(self, parts):
        self.parts = sorted((p for p in parts if p.suffix in (".zip",) or p.suffix.startswith(".z0")), key=lambda p: (p.suffix == ".zip", p.name))
        self.parts = sorted(self.parts, key=lambda p: p.name)  # sort by name
        # Move .zip to end
        self.parts = [p for p in self.parts if p.suffix != ".zip"] + [p for p in self.parts if p.suffix == ".zip"]
        self._sizes = [p.stat().st_size for p in self.parts]
        self._offsets = []
        off = 0
        for sz in self._sizes:
            self._offsets.append(off)
            off += sz
        self._total = off
        self._pos = 0
        self._fh = None
        self._cur_idx = -1

    def _ensure_open(self, idx):
        if self._cur_idx != idx:
            self._close()
            self._fh = open(self.parts[idx], "rb")
            self._cur_idx = idx

    def _close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
            self._cur_idx = -1

    def readable(self):
        return True

    def readinto(self, b):
        n = len(b)
        remaining = self._total - self._pos
        if remaining <= 0:
            return 0
        to_read = min(n, remaining)
        orig = to_read
        buf = b
        while to_read > 0:
            idx = next(i for i, off in enumerate(self._offsets) if i == len(self._offsets) - 1 or self._pos < self._offsets[i + 1])
            self._ensure_open(idx)
            self._fh.seek(self._pos - self._offsets[idx])
            chunk = self._fh.read(to_read)
            if not chunk:
                break
            buf[orig - to_read: orig - to_read + len(chunk)] = chunk
            to_read -= len(chunk)
            self._pos += len(chunk)
        self._close()
        return orig - to_read

    def seek(self, offset, whence=0):
        if whence == 0: self._pos = offset
        elif whence == 1: self._pos += offset
        else: self._pos = self._total + offset
        self._pos = max(0, min(self._pos, self._total))
        return self._pos

    def tell(self):
        return self._pos

    def close(self):
        self._close()


def download_file(file_id, out_path):
    """Download a single file. Returns True on success."""
    import gdown
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".part"
    try:
        gdown.download(id=file_id, output=tmp, quiet=False, resume=True)
        if Path(tmp).exists():
            if out_path.exists():
                out_path.unlink()
            shutil.move(tmp, out_path)
            return True
        return False
    except Exception as e:
        print(f"  Error: {e}")
        if Path(tmp).exists():
            Path(tmp).unlink()
        return False


def extract_split_zip(part_dir, gen_name, out_dir):
    """Extract GenImage split zip to out_dir."""
    parts = sorted(part_dir.glob(f"*{gen_name.lower()}*")) + sorted(part_dir.glob(f"imagenet_*"))
    parts = [p for p in parts if p.suffix in (".zip", ".z01", ".z02", ".z03", ".z04", ".z05", ".z06", ".z07", ".z08", ".z09", ".z10", ".z11", ".z12", ".z13")]
    parts = sorted(set(p for p in parts if p.exists()))
    
    if not parts:
        print(f"  No zip parts found matching {gen_name} in {part_dir}")
        return False
    
    print(f"  Found {len(parts)} zip parts")
    reader = SplitZipReader(parts)
    try:
        with ZipFile(reader) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            total = len(names)
            print(f"  Archive contains {total} files")
            
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            for i, name in enumerate(names):
                ext = Path(name).suffix.lower()
                if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                    continue
                out_path = out_dir / name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(zf.read(name))
                if (i + 1) % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(f"    Extracted {i+1}/{total} ({rate:.0f} img/s)...")
            
            elapsed = time.time() - t0
            print(f"  Extracted {total} files in {elapsed:.0f}s ({total/elapsed:.0f} img/s)")
            return True
    except Exception as e:
        print(f"  Extract failed: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", choices=list(GENERATORS.keys()), required=True)
    parser.add_argument("--download-dir", type=str, default="genimage_zips")
    args = parser.parse_args()

    gen = args.generator
    files = GENERATORS[gen]
    dl_dir = Path(args.download_dir)
    dl_dir.mkdir(parents=True, exist_ok=True)

    gen_out = GENIMAGE_OUT / gen
    gen_out.mkdir(parents=True, exist_ok=True)

    print(f"Generator: {gen} ({len(files)} parts)")
    print(f"Download to: {dl_dir.resolve()}")
    print(f"Extract to: {gen_out.resolve()}")
    print()

    # Download
    t_start = time.time()
    for i, (fid, fname) in enumerate(files):
        out_path = dl_dir / fname
        if out_path.exists():
            print(f"[{i+1}/{len(files)}] {fname} — already exists ({fmt_bytes(out_path.stat().st_size)})")
            continue
        
        print(f"[{i+1}/{len(files)}] Downloading {fname}...")
        t0 = time.time()
        ok = download_file(fid, out_path)
        if not ok:
            print(f"  FAILED: {fname}")
            continue
        elapsed = time.time() - t0
        sz = out_path.stat().st_size
        speed = sz / elapsed / 1e6 if elapsed > 0 else 0
        print(f"  Done: {fmt_bytes(sz)} in {elapsed:.0f}s ({speed:.1f} MB/s)")

    total_elapsed = time.time() - t_start
    downloaded = sum(1 for _, fn in files if (dl_dir / fn).exists())
    print(f"\nDownloaded {downloaded}/{len(files)} parts in {total_elapsed:.0f}s")

    # Extract
    if downloaded == len(files):
        print(f"\nExtracting {gen}...")
        if extract_split_zip(dl_dir, gen, gen_out):
            print(f"Extraction complete! Images in: {gen_out}")
            # Clean up zip parts
            ans = input("\nDelete zip parts? (y/n): ")
            if ans.lower() == "y":
                for _, fn in files:
                    p = dl_dir / fn
                    if p.exists():
                        p.unlink()
                print("Zip parts deleted")
    else:
        print(f"\nNot all parts downloaded ({downloaded}/{len(files)}). Cannot extract.")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
