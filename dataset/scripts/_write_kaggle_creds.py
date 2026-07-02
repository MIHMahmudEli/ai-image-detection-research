"""Write Kaggle credentials"""
import json
from pathlib import Path

kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(exist_ok=True)

creds = {"username": "rowtech", "key": "KGAT_100897ec213fc73a81649eb3d20cea99"}
kf = kaggle_dir / "kaggle.json"
kf.write_text(json.dumps(creds))
# Set permissions (Windows doesn't need chmod)
print(f"Written to {kf}")
