"""Test SplitZipReader with mock data."""
import io, zipfile, os, tempfile, shutil
from pathlib import Path


class SplitZipReader(io.RawIOBase):
    def __init__(self, dir_path):
        self.dir = Path(dir_path)
        parts = sorted(self.dir.glob("*.z[0-9][0-9]")) + sorted(self.dir.glob("*.zip"))
        if not parts:
            raise FileNotFoundError(f"No zip parts in {dir_path}")
        self.parts = parts
        self._sizes = [p.stat().st_size for p in parts]
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
            idx = next(
                i
                for i, off in enumerate(self._offsets)
                if i == len(self._offsets) - 1 or self._pos < self._offsets[i + 1]
            )
            self._ensure_open(idx)
            self._fh.seek(self._pos - self._offsets[idx])
            chunk = self._fh.read(to_read)
            if not chunk:
                break
            view = memoryview(buf)
            view[orig - to_read : orig - to_read + len(chunk)] = chunk
            to_read -= len(chunk)
            self._pos += len(chunk)
        self._close()
        return orig - to_read

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        else:
            self._pos = self._total + offset
        self._pos = max(0, min(self._pos, self._total))
        return self._pos

    def tell(self):
        return self._pos

    def close(self):
        self._close()


def test():
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "hello.txt").write_text("hello" * 100000)  # 500KB
        (tmp / "world.txt").write_text("world" * 100000)  # 500KB

        zip_path = tmp / "test.zip"
        with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_STORED) as zf:
            zf.write(str(tmp / "hello.txt"), "hello.txt")
            zf.write(str(tmp / "world.txt"), "world.txt")

        data = zip_path.read_bytes()
        print(f"Total zip size: {len(data)} bytes")
        zip_path.unlink()

        # Write as split parts (split at 300000 and 600000)
        (tmp / "test.z01").write_bytes(data[:300000])
        (tmp / "test.z02").write_bytes(data[300000:600000])
        (tmp / "test.zip").write_bytes(data[600000:])

        for fn in ["test.z01", "test.z02", "test.zip"]:
            p = tmp / fn
            sz = p.stat().st_size if p.exists() else 0
            print(f"  {fn}: {sz} bytes")

        reader = SplitZipReader(tmp)
        with zipfile.ZipFile(reader) as zf:
            names = zf.namelist()
            print(f"Files in archive: {names}")
            for n in names:
                print(f"  {n}: {len(zf.read(n))} bytes")
        reader.close()
        print("TEST PASSED")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test()
