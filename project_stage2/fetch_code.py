"""Fetch training code from Jupyter server"""
import requests, re, json, os

jupyter_url = os.environ["METEORITE_JUPYTER_URL"].rstrip("/")
session = requests.Session()
r = session.get(f'{jupyter_url}/login?next=%2Flab', timeout=10)
match = re.search(r'name="_xsrf" value="([^"]+)"', r.text)
session.post(f'{jupyter_url}/login?next=%2Flab',
    data={'password': os.environ["METEORITE_JUPYTER_PASSWORD"], '_xsrf': match.group(1)}, timeout=10)

scripts = [
    'final%20project/external/teammate_repo/train_timm_frozen.py',
    'final%20project/external/teammate_repo/infer_tta_dinov3.py',
    'final%20project/external/teammate_repo/main.py',
    'final%20project/external/teammate_repo/utils.py',
    'final%20project/external/teammate_repo/hhp_main_f1.py',
]

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'teammate_code')
os.makedirs(out_dir, exist_ok=True)

for script_path in scripts:
    fname = script_path.split('/')[-1]
    r = session.get(f'{jupyter_url}/api/contents/{script_path}', timeout=10)
    if r.status_code != 200:
        print(f'Failed: {script_path} ({r.status_code})')
        continue

    data = r.json()
    content = data.get('content', '')

    # Extract source from notebook cells
    full_source = []
    if isinstance(content, dict) and 'cells' in content:
        nb = content
        for cell in nb.get('cells', []):
            src = cell.get('source', '')
            if isinstance(src, list):
                src = ''.join(src)
            full_source.append(src)
        text = '\n'.join(full_source)
    elif isinstance(content, str):
        text = content
    else:
        print(f'Unknown format for {fname}: {type(content)}')
        continue

    local_path = os.path.join(out_dir, fname)
    with open(local_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'Saved: {fname} ({len(text)} chars)')

print(f'\nFiles in: {out_dir}')
