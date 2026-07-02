"""Check GenImage dataset download links"""
import requests, re

r = requests.get("https://genimage-dataset.github.io/", timeout=15)
html = r.text
links = re.findall(r'href=[\'"](.*?)[\'"]', html)
for link in links:
    lower = link.lower()
    if any(x in lower for x in ["download", "zip", "tar", "gz", "dataset", "data", "google", "drive", "github"]):
        print(link)
