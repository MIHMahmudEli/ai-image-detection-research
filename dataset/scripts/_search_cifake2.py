"""Search for CIFAKE on Kaggle"""
import requests

headers = {"User-Agent": "Mozilla/5.0"}
r = requests.get("https://www.kaggle.com/api/v1/datasets/search?search=cifake", headers=headers, timeout=15)
print(f"Search: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    for res in data.get("results", [])[:10]:
        print(f"  {res.get('ref')} - {res.get('title')}")
else:
    # Try web scraping
    r2 = requests.get("https://www.kaggle.com/search?q=cifake", headers=headers, timeout=15)
    print(f"Web search: {r2.status_code}")
    # Check if CIFAKE appears in results
    if "cifake" in r2.text.lower():
        print("Found CIFAKE reference in search results")
    import re
    refs = re.findall(r'/datasets/([^/\"]+/[^/\"]+)', r2.text)
    for ref in set(refs):
        print(f"  kaggle.com/datasets/{ref}")
