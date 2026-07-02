"""
Extract GenImage split zip archives (.z01, .z02, ..., .zip) by directly
parsing PKZIP format. Works without 7-Zip.
"""
import struct, os, sys, time
from pathlib import Path


def read_all(parts):
    """Generator that yields all file data in order."""
    for p in parts:
        with open(p, "rb") as f:
            while True:
                chunk = f.read(8192 * 1024)
                if not chunk:
                    break
                yield chunk


def find_eocd(data, size):
    """Find End of Central Directory record (PK\x05\x06)."""
    for i in range(size - 22, -1, -1):
        if data[i:i+4] == b"PK\x05\x06":
            return i
    return -1


def parse_eocd(data, pos):
    """Parse EOCD record and look for Zip64 EOCD."""
    sig, disk_num, disk_cd, num_entries_disk, num_entries_total, cd_size, cd_offset, comment_len = struct.unpack_from("<IHHHHIIH", data, pos)
    
    result = {
        "disk_num": disk_num,
        "disk_cd": disk_cd,
        "num_entries": num_entries_total,
        "cd_size": cd_size,
        "cd_offset": cd_offset,
        "comment_len": comment_len,
    }
    
    # Check for Zip64 EOCD locator (PK\x06\x07) right before EOCD
    if pos >= 20:
        loc_sig = struct.unpack_from("<I", data, pos - 20)[0]
        if loc_sig == 0x07064b50:
            _, disk_with_zeocd, zeocd_offset, total_disks = struct.unpack_from(
                "<IIQI", data, pos - 16)
            result["zip64_offset"] = zeocd_offset
            result["zip64_disk"] = disk_with_zeocd
    return result


def parse_central_dir_entry(data, pos):
    """Parse a central directory entry."""
    sig = struct.unpack_from("<I", data, pos)[0]
    if sig != 0x02014b50:
        return None, pos
    (made_ver, needed_ver, flags, method, modtime, moddate, crc32,
     comp_size, uncomp_size, name_len, extra_len, comment_len,
     disk_start, int_attr, ext_attr, local_offset) = struct.unpack_from(
        "<HHHHHHIIIHHHHHII", data, pos + 4)
    
    name = data[pos + 46:pos + 46 + name_len].decode("utf-8", errors="replace")
    
    entry = {
        "name": name,
        "method": method,
        "crc32": crc32,
        "comp_size": comp_size,
        "uncomp_size": uncomp_size,
        "local_offset": local_offset,
        "disk_start": disk_start,
        "name_len": name_len,
        "extra_len": extra_len,
    }
    return entry, pos + 46 + name_len + extra_len + comment_len


def extract_from_stream(parts, entry, out_dir):
    """Extract a single file entry from the zipped stream."""
    # local_offset is relative to the disk specified by disk_start
    disk_start = entry["disk_start"]
    local_offset = entry["local_offset"]
    
    if disk_start >= len(parts):
        print(f"  Invalid disk_start {disk_start} for {entry['name']}")
        return False
    
    part_file = parts[disk_start]
    fh = open(part_file, "rb")
    fh.seek(local_offset)
    
    # Parse local file header
    header = fh.read(30)
    sig = struct.unpack_from("<I", header, 0)[0]
    if sig != 0x04034b50:
        fh.close()
        print(f"  Invalid local header at offset {local_offset}")
        return False
    
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    fh.read(name_len + extra_len)  # skip filename + extra
    
    # Read compressed data
    comp_size = entry["comp_size"]
    if comp_size == 0:
        fh.close()
        return True  # directory
    
    data = bytearray()
    remaining = comp_size
    while remaining > 0:
        chunk = fh.read(min(remaining, 8192 * 1024))
        if not chunk:
            break
        data.extend(chunk)
        remaining -= len(chunk)
    
    fh.close()
    
    # Decompress if needed
    raw_data = bytes(data)
    if entry["method"] == 0:
        # Stored
        result = raw_data
    elif entry["method"] == 8:
        # Deflated
        import zlib
        try:
            result = zlib.decompress(raw_data, -zlib.MAX_WBITS)
        except:
            try:
                result = zlib.decompress(raw_data)
            except Exception as e:
                print(f"  Decompress failed for {entry['name']}: {e}")
                return False
    else:
        print(f"  Unsupported compression method {entry['method']} for {entry['name']}")
        return False
    
    out_path = out_dir / entry["name"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(result)
    return True


def extract_split_zip(part_dir, gen_name, out_dir):
    """Main extract function."""
    parts = sorted(p for p in Path(part_dir).iterdir() 
                   if p.suffix in (".zip", ".z01", ".z02", ".z03", ".z04", 
                                   ".z05", ".z06", ".z07", ".z08", ".z09",
                                   ".z10", ".z11", ".z12", ".z13", ".z14"))
    # Sort by name (z01 before z02 before zip)
    parts = sorted(parts, key=lambda p: p.name)
    # Make sure .zip is last
    zip_part = [p for p in parts if p.suffix == ".zip"]
    other_parts = [p for p in parts if p.suffix != ".zip"]
    parts = other_parts + zip_part
    
    if not parts:
        print(f"  No zip parts found in {part_dir}")
        return False
    
    print(f"  Found {len(parts)} parts: {[p.name for p in parts]}")
    
    # Read last 64KB of last file to find EOCD
    last = parts[-1]
    chunk_size = min(last.stat().st_size, 65536)
    with open(last, "rb") as f:
        f.seek(-chunk_size, 2)
        tail = f.read(chunk_size)
    
    eocd_pos = find_eocd(tail, chunk_size)
    if eocd_pos < 0:
        print(f"  Cannot find EOCD record in last part")
        return False
    
    eocd = parse_eocd(tail, eocd_pos)
    
    # Check for Zip64 EOCD for actual entry count
    num_entries = eocd["num_entries"]
    cd_offset = eocd["cd_offset"]
    cd_size = eocd["cd_size"]
    disk_cd = eocd["disk_cd"]
    
    if "zip64_offset" in eocd and num_entries == 0xFFFF:
        # Read Zip64 EOCD record for actual values
        z64_disk = eocd["zip64_disk"]
        z64_offset = eocd["zip64_offset"]
        z64_part = parts[z64_disk] if z64_disk < len(parts) else parts[-1]
        with open(z64_part, "rb") as f:
            f.seek(z64_offset)
            z64_data = f.read(56)
        z64_sig = struct.unpack_from("<I", z64_data, 0)[0]
        if z64_sig == 0x06064b50:
            (_, z64_size, _, _, _, _,
             num_entries_disk_z64, num_entries_z64,
             cd_size_z64, cd_offset_z64) = struct.unpack_from(
                "<IQHHIIQQQQ", z64_data, 0)
            num_entries = num_entries_z64
            cd_size = cd_size_z64
            cd_offset = cd_offset_z64
            print(f"  Zip64 EOCD: {num_entries} entries, "
                  f"offset={cd_offset}, size={cd_size}")
    
    print(f"  Central directory: {num_entries} entries, "
          f"offset_in_disk={cd_offset}, size={cd_size}")
    
    # Use the disk_cd-th part as the base for the cd_offset
    cd_part = parts[disk_cd] if disk_cd < len(parts) else parts[-1]
    print(f"  Central directory on disk {disk_cd} ({cd_part.name})")
    
    cd_data = bytearray()
    with open(cd_part, "rb") as f:
        f.seek(cd_offset)
        remaining = cd_size
        while remaining > 0:
            chunk = f.read(min(remaining, 8192 * 1024))
            if not chunk:
                break
            cd_data.extend(chunk)
            remaining -= len(chunk)
    
    # Parse central directory entries
    entries = []
    pos = 0
    parsed = 0
    while pos < len(cd_data) and parsed < num_entries:
        entry, pos = parse_central_dir_entry(bytes(cd_data), pos)
        if entry is None:
            break
        parsed += 1
        if not entry["name"].endswith("/"):  # skip directories
            entries.append(entry)
    
    print(f"  Parsed {len(entries)} file entries (stopped at entry {parsed})")
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    success = 0
    for i, entry in enumerate(entries):
        ext = Path(entry["name"]).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        if extract_from_stream(parts, entry, out_dir):
            success += 1
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    Extracted {i+1}/{len(entries)} ({success} images, "
                  f"{success/elapsed:.0f} img/s)...")
    
    elapsed = time.time() - t0
    print(f"  Extracted {success} images in {elapsed:.0f}s "
          f"({success/elapsed:.0f} img/s)")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_split_zip.py <part_dir> [out_dir]")
        sys.exit(1)
    
    part_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else part_dir.parent / "extracted"
    
    gen_name = part_dir.name if part_dir.is_dir() else ""
    extract_split_zip(part_dir, gen_name, out_dir)
