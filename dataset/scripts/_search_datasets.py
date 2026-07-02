"""Search for direct download sources for AI image datasets"""
import requests, re

# Check multiple sources
datasets = [
    "https://raw.githubusercontent.com/sagnik1511/CIFAKE/main/README.md",
    "https://huggingface.co/datasets/Falconsai/ai_image_detection",
    "https://huggingface.co/datasets/imatag/ai-generated-image-detection",
]

for url in datasets:
    try:
        r = requests.get(url, timeout=10)
        print(f"{url}: HTTP {r.status_code} ({len(r.content)/1024:.0f}KB)")
        if r.status_code == 200:
            text = r.text
            links = re.findall(r'(https?://[^\s\)\"<>]+)', text)
            for link in links[:5]:
                if any(x in link for x in ["google", "drive", "zip", "download"]):
                    print(f"  -> {link}")
    except Exception as e:
        print(f"{url}: ERROR {str(e)[:40]}")
