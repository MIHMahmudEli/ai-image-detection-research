"""
List Active Runs
=================
Local, cross-run status dashboard via HuggingFace API.

Enumerates all runs/*/logs/training_state.json in the checkpoint repo,
downloads each, prints a status table showing:
  run_id | model_variant | current_epoch | best_val_macro_f1 | status | manifest_sha256 (short) | last_updated

Flags any run whose manifest_sha256 doesn't match the current
top-level manifest — the check that confirms all parallel runs
are still comparable.

Usage:
    python list_active_runs.py

Requires:
    .env file with HF_TOKEN
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_config import HF_CHECKPOINT_REPO, HF_MANIFEST_REPO, MANIFEST_PATH_IN_REPO


def get_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("hf="):
                    return line.strip().split("=", 1)[1]
    raise ValueError("HF_TOKEN not found in .env")


def compute_manifest_sha256(manifest: dict) -> str:
    m = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()


def main():
    hf_token = get_hf_token()

    print("=" * 90)
    print("MFFT Active Runs Dashboard")
    print("=" * 90)

    # 1. Download current manifest for comparison
    print("\nDownloading current manifest...")
    current_manifest_sha256 = None
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_MANIFEST_REPO,
            filename=MANIFEST_PATH_IN_REPO,
            repo_type="model",
            token=hf_token,
        )
        with open(path) as f:
            manifest = json.load(f)
        current_manifest_sha256 = compute_manifest_sha256(manifest)
        print(f"  Current manifest SHA-256: {current_manifest_sha256[:16]}...")
        print(f"  Version: {manifest.get('version', '?')}")
        print(f"  Images: {manifest.get('total_images', '?')}")
    except Exception as e:
        print(f"  WARNING: Could not download manifest: {e}")

    # 2. List all files in the repo to find runs
    print("\nScanning checkpoint repo for runs...")
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)

    try:
        files = list(api.list_repo_files(repo_id=HF_CHECKPOINT_REPO, repo_type="model"))
    except Exception as e:
        print(f"  ERROR listing repo: {e}")
        print("  (Repo may be empty or not exist yet)")
        files = []

    # Find all training_state.json files
    state_files = [f for f in files if "/logs/training_state.json" in f]

    if not state_files:
        print("\n  No runs found in checkpoint repo.")
        print(f"  Repo: https://huggingface.co/{HF_CHECKPOINT_REPO}")
        return

    print(f"\n  Found {len(state_files)} run(s)")

    # 3. Download and display each state
    print(f"\n{'─'*90}")
    header = f"{'run_id':<35} {'epoch':>5} {'best_f1':>8} {'status':<15} {'manifest_sha256':<18} {'last_updated':<12}"
    print(header)
    print(f"{'─'*90}")

    runs_data = []
    for state_file in state_files:
        # Extract run_id from path: runs/<run_id>/logs/training_state.json
        parts = state_file.split("/")
        if len(parts) < 4:
            continue
        run_id = parts[1]

        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id=HF_CHECKPOINT_REPO,
                filename=state_file,
                repo_type="model",
                token=hf_token,
            )
            with open(local_path) as f:
                state = json.load(f)

            run_manifest_hash = state.get("manifest_sha256", "")
            hash_match = (
                current_manifest_sha256 is None
                or not run_manifest_hash
                or run_manifest_hash == current_manifest_sha256
            )

            runs_data.append({
                "run_id": run_id,
                "model_variant": state.get("model_variant", "?"),
                "current_epoch": state.get("current_epoch", 0),
                "best_val_macro_f1": state.get("best_val_macro_f1", 0),
                "status": state.get("status", "?"),
                "manifest_sha256": run_manifest_hash[:16] if run_manifest_hash else "N/A",
                "manifest_match": hash_match,
                "last_updated": state.get("last_updated", "")[:10],
                "manifest_version": state.get("manifest_version", "?"),
            })

            # Highlight runs with mismatched manifest
            flag = "" if hash_match else " *** MANIFEST MISMATCH"
            print(
                f"{run_id:<35} {state.get('current_epoch', 0):>5} "
                f"{state.get('best_val_macro_f1', 0):>8.4f} "
                f"{state.get('status', '?'):<15} "
                f"{run_manifest_hash[:16] if run_manifest_hash else 'N/A':<18} "
                f"{state.get('last_updated', '')[:10]:<12}{flag}"
            )

        except Exception as e:
            print(f"{run_id:<35} ERROR: {e}")

    print(f"{'─'*90}")

    # 4. Summary
    if runs_data:
        print(f"\nSummary:")
        print(f"  Total runs: {len(runs_data)}")
        statuses = {}
        for r in runs_data:
            s = r["status"]
            statuses[s] = statuses.get(s, 0) + 1
        for s, count in statuses.items():
            print(f"    {s}: {count}")

        mismatches = [r for r in runs_data if not r["manifest_match"]]
        if mismatches:
            print(f"\n  WARNING: {len(mismatches)} run(s) have manifest SHA-256 mismatch:")
            for r in mismatches:
                print(f"    {r['run_id']}: {r['manifest_sha256']}")
            print(f"  These runs evaluated on DIFFERENT data splits and are NOT comparable.")

        # Best model
        best = max(runs_data, key=lambda r: r["best_val_macro_f1"])
        print(f"\n  Best model: {best['run_id']} (val_macro_f1={best['best_val_macro_f1']:.4f})")

    print(f"\nCheckpoint repo: https://huggingface.co/{HF_CHECKPOINT_REPO}")


if __name__ == "__main__":
    main()
