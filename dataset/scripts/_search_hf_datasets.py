"""Find accessible AI image detection datasets on HuggingFace"""
import requests
import re

# Check known datasets
datasets = [
    "Falconsai/ai_image_detection",
    "imatag/ai-generated-image-detection", 
    "sp1187/cifake",
    "sasha/aigen_images",
]

for ds in datasets:
    r = requests.head(f"https://huggingface.co/datasets/{ds}", timeout=10)
    size = r.headers.get("content-length", "?")
    print(f"{ds:45s} -> {r.status_code} ({r.headers.get('content-type','?')[:30]})")
