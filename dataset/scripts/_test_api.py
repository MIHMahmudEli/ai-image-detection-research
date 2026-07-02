import requests
resp = requests.get('https://civitai.com/api/v1/images?limit=5&page=1', timeout=15)
print(f'Status: {resp.status_code}')
data = resp.json()
items = data.get('items', [])
print(f'Items: {len(items)}')
if items:
    print(f'First item url: {str(items[0].get("url", ""))[:100]}')
