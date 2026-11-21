"""Deploy the proper v13 training pipeline to server"""
import paramiko
import os
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
    print('Connected.')
    return client


def run_cmd(client, cmd, timeout=7200):
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
        print(f'Uploaded: {os.path.basename(local_path)}')
    finally:
        sftp.close()


def download_dir(client, remote_dir, local_dir):
    import stat
    os.makedirs(local_dir, exist_ok=True)
    sftp = client.open_sftp()
    try:
        for item in sftp.listdir_attr(remote_dir):
            rp = f'{remote_dir}/{item.filename}'
            lp = os.path.join(local_dir, item.filename)
            if stat.S_ISDIR(item.st_mode):
                download_dir(client, rp, lp)
            else:
                try:
                    sftp.get(rp, lp)
                    print(f'  {item.filename}')
                except:
                    pass
    finally:
        sftp.close()


def main():
    client = connect()

    # Step 1: Check env and data
    print('\n' + '=' * 60)
    print('Step 1: Environment check')
    print('=' * 60)

    checks = [
        'nvidia-smi --query-gpu=name,memory.free --format=csv,noheader',
        'ls /data/models/dinov3_large/model.safetensors && echo "DINOv3: OK"',
        'ls /data/models/convnextv2_base_22k_384_ema.pt && echo "ConvNeXtV2-Base: OK"',
        'ls /data/数据科学实践project/train_labels.csv && echo "Train CSV: OK"',
        'ls /data/数据科学实践project/test_images/ | head -3',
        'python -c "import timm; print(f\"timm: {timm.__version__}\")" 2>&1',
    ]
    for cmd in checks:
        stdin, stdout, stderr = client.exec_command(cmd, timeout=15)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if out:
            print(f'  {out[:200]}')
        if err:
            print(f'  [ERR] {err[:200]}')

    # Step 2: Upload training script
    print('\n' + '=' * 60)
    print('Step 2: Upload train_v13_proper.py')
    print('=' * 60)
    upload_file(client,
        os.path.join(LOCAL_DIR, 'train_v13_proper.py'),
        f'{SERVER_PROJECT}/train_v13_proper.py')

    # Step 3: Run training pipeline
    print('\n' + '=' * 60)
    print(f'Step 3: Run training (start: {time.strftime("%H:%M:%S")})')
    print('=' * 60)
    run_cmd(client, f'cd {SERVER_PROJECT} && python train_v13_proper.py 2>&1', timeout=7200)

    # Step 4: Download results
    print('\n' + '=' * 60)
    print('Step 4: Download results')
    print('=' * 60)

    stdin, stdout, stderr = client.exec_command(
        f'ls -d {SERVER_PROJECT}/runs/v13_proper 2>/dev/null', timeout=10)
    base_dir = stdout.read().decode().strip()
    if base_dir:
        local_out = os.path.join(LOCAL_DIR, 'v13_proper_output')
        download_dir(client, base_dir, local_out)
        print(f'\nDownloaded to: {local_out}')

        # List all submission files
        print('\n  Submission files:')
        for root, dirs, files in os.walk(local_out):
            for f in files:
                if f.endswith('.csv'):
                    print(f'    {os.path.relpath(os.path.join(root, f), local_out)}')
    else:
        print('No output found!')

    client.close()
    print('\nDone!')


if __name__ == '__main__':
    main()
