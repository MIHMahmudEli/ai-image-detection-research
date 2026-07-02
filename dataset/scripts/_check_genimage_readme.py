"""Get GenImage download links from README"""
import requests, re

r = requests.get("https://raw.githubusercontent.com/GenImage-Dataset/GenImage/main/Readme.md", timeout=15)
text = r.text

# Find all URLs
urls = re.findall(r'(https?://[^\s\)\"]+)', text)
for u in urls:
    lower = u.lower()
    if any(x in lower for x in ["google", "drive", "download", "zip", "data"]):
        print(u)

print()
print(text[:3000])
