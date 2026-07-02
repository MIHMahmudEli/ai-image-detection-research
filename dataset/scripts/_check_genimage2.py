"""Check GenImage GitHub repo and README"""
import requests

r = requests.get("https://api.github.com/repos/GenImage-Dataset/GenImage", timeout=15)
if r.status_code != 200:
    print(f"API: {r.status_code}")
    exit()

d = r.json()
print(f"Stars: {d.get('stargazers_count')}")
print(f"URL: {d.get('html_url')}")
print()

# Get README for download instructions
r2 = requests.get("https://raw.githubusercontent.com/GenImage-Dataset/GenImage/main/README.md", timeout=15)
if r2.status_code == 200:
    text = r2.text
    # Find URLs in README
    import re
    urls = re.findall(r'(https?://[^\s\)]+)', text)
    for url in urls:
        if any(x in url for x in ["google", "drive", "download", "zip", "data"]):
            print(url)
