"""Deploy v13 blend to server and run it"""
import paramiko
import os
import sys
import time

HOST = os.environ["METEORITE_SSH_HOST"]
PORT = int(os.environ.get("METEORITE_SSH_PORT", "22"))
USER = os.environ["METEORITE_SSH_USER"]
PASSWORD = os.environ["METEORITE_SSH_PASSWORD"]

SERVER_PROJECT = '/data/数据科学实践project'
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
    print('Connected to server.')
    return client


def run_cmd(client, cmd, timeout=1800):
    """Run command, stream output line by line."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    for line in iter(stdout.readline, ''):
        if line:
            print(line.rstrip())
    err = stderr.read().decode().strip()
    if err:
        print(f'[STDERR]: {err[:1000]}')


def upload_file(client, local_path, remote_path):
    sftp = client.open_sftp()
    try:
        sftp.put(local_path, remote_path)
        print(f'Uploaded: {os.path.basename(local_path)} -> {remote_path}')
    finally:
        sftp.close()


def download_dir(client, remote_dir, local_dir):
    """Download all files from a remote directory."""
    import stat
    os.makedirs(local_dir, exist_ok=True)
    sftp = client.open_sftp()
    try:
        for item in sftp.listdir_attr(remote_dir):
            remote_path = f'{remote_dir}/{item.filename}'
            local_path = os.path.join(local_dir, item.filename)
            if stat.S_ISDIR(item.st_mode):
                download_dir(client, remote_path, local_path)
            else:
                try:
                    sftp.get(remote_path, local_path)
                    print(f'  Downloaded: {item.filename}')
                except Exception as e:
                    print(f'  Failed: {item.filename} - {e}')
    finally:
        sftp.close()


def main():
    client = connect()

    # Step 1: Check environment
    print('\n' + '=' * 60)
    print('Step 1: Check server environment')
    print('=' * 60)

    checks = [
        'echo "Host: $(hostname)"',
        'nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU"',
        'ls /data/数据科学实践project/train_labels.csv 2>/dev/null && echo "Data OK" || echo "Data NOT FOUND"',
        'python -c "import timm; print(f\"timm: {timm.__version__}\")" 2>&1',
        'python -c "import torch; print(f\"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}\")" 2>&1',
        'python -c "from safetensors.torch import load_file; print(\"safetensors: OK\")" 2>&1',
        'pip list 2>/dev/null | grep -iE "sklearn|scikit" || echo "sklearn NOT FOUND"',
    ]
    for cmd in checks:
        stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out:
            print(f'  {out}')
        if err:
            print(f'  [ERR] {err[:200]}')

    # Step 2: Install missing packages if needed
    print('\n' + '=' * 60)
    print('Step 2: Install dependencies')
    print('=' * 60)
    run_cmd(client, 'pip install scikit-learn safetensors timm pandas tqdm pillow -q 2>&1 | tail -5', timeout=120)

    # Step 3: Upload v13 blend script
    print('\n' + '=' * 60)
    print('Step 3: Upload train_v13_blend.py')
    print('=' * 60)
    local_script = os.path.join(LOCAL_DIR, 'train_v13_blend.py')
    remote_script = f'{SERVER_PROJECT}/train_v13_blend.py'
    upload_file(client, local_script, remote_script)

    # Step 4: Run the script
    print('\n' + '=' * 60)
    print(f'Step 4: Run train_v13_blend.py (start: {time.strftime("%H:%M:%S")})')
    print('=' * 60)
    run_cmd(client, f'cd {SERVER_PROJECT} && python train_v13_blend.py 2>&1', timeout=3600)

    # Step 5: Download results
    print('\n' + '=' * 60)
    print('Step 5: Download results')
    print('=' * 60)

    # Find the latest output dir
    stdin, stdout, stderr = client.exec_command(
        f'ls -d {SERVER_PROJECT}/runs/v13_blend_* 2>/dev/null | sort | tail -1',
        timeout=15
    )
    latest_dir = stdout.read().decode().strip()
    if latest_dir:
        print(f'Latest output: {latest_dir}')
        local_out = os.path.join(LOCAL_DIR, 'v13_blend_output')
        download_dir(client, latest_dir, local_out)
        print(f'\nResults downloaded to: {local_out}')
    else:
        print('No output directory found!')

    client.close()
    print('\nDone!')


if __name__ == '__main__':
    main()
