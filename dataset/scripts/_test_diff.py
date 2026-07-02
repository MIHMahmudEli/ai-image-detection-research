import requests
r = requests.head("https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part_00000.zip", timeout=15)
print(f"Status: {r.status_code}")
print(f"Content-Length: {r.headers.get('content-length', 'N/A')}")
