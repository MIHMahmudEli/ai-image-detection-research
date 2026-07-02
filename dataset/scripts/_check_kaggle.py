"""Check Kaggle credential status"""
import os
from pathlib import Path

print("KAGGLE_USERNAME:", os.environ.get("KAGGLE_USERNAME", "NOT SET"))
print("KAGGLE_KEY:", "SET" if os.environ.get("KAGGLE_KEY") else "NOT SET")

kf = Path.home() / ".kaggle" / "kaggle.json"
print("kaggle.json:", "EXISTS" if kf.exists() else "NOT FOUND")
if kf.exists():
    print("  Location:", kf)
