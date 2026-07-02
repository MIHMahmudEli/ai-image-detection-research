import requests

repos = [
    ('GenImage_BigGAN', 'BigGAN'),
    ('GenImage_ADM', 'ADM'),
    ('GenImage_MidJourney', 'MidJourney'),
    ('GenImage_glide', 'glide'),
    ('GenImage_wukong', 'wukong'),
    ('GenImage_VQDM', 'VQDM'),
    ('GenImage_SD14', 'stable_diffusion_v_1_4'),
    ('GenImage_SD15', 'stable_diffusion_v_1_5'),
]

for repo, name in repos:
    url = f'https://huggingface.co/api/datasets/bitmind/{repo}'
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        siblings = data.get('siblings', [])
        shards = [s['rfilename'] for s in siblings if s['rfilename'].endswith('.parquet')]
        total_downloads = data.get('downloads', '?')
        print(f'bitmind/{repo}: {len(shards)} shards, {total_downloads} downloads')
    else:
        print(f'bitmind/{repo}: status {r.status_code} - trying bitmind/GenImage_{name}...')
        # Try with generator name
        url2 = f'https://huggingface.co/api/datasets/bitmind/GenImage_{name}'
        r2 = requests.get(url2)
        if r2.status_code == 200:
            data = r2.json()
            siblings = data.get('siblings', [])
            shards = [s['rfilename'] for s in siblings if s['rfilename'].endswith('.parquet')]
            print(f'  bitmind/GenImage_{name}: {len(shards)} shards')
        else:
            print(f'  Not found either')
