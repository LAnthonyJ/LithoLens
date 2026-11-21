"""
Deploy and run zero-shot meteorite classification on GPU server
"""
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

def run_cmd(client, cmd, timeout=120):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    return out, err

def upload_file(client, local_path, remote_path):
    sftp = client.open_sftp()
    try:
        sftp.put(local_path, remote_path)
        print(f'  Uploaded: {os.path.basename(local_path)} -> {remote_path}')
    finally:
        sftp.close()

def download_file(client, remote_path, local_path):
    sftp = client.open_sftp()
    try:
        sftp.get(remote_path, local_path)
        print(f'  Downloaded: {remote_path} -> {local_path}')
    finally:
        sftp.close()

def main():
    client = connect()

    # Step 1: Upload inference script
    print('\n=== Step 1: Upload Script ===')
    local_script = os.path.join(LOCAL_DIR, 'server_zero_shot.py')
    remote_script = f'{SERVER_PROJECT}/server_zero_shot.py'
    upload_file(client, local_script, remote_script)

    # Step 2: Check HF access
    print('\n=== Step 2: Check HF Access ===')
    out, err = run_cmd(client, f'cd {SERVER_PROJECT} && HF_ENDPOINT=https://hf-mirror.com python3 -c "from huggingface_hub import HfApi; api=HfApi(); print(api.model_info(\"openai/clip-vit-large-patch14\", timeout=15).modelId)" 2>&1', timeout=30)
    print(f'HF Mirror: {out[:300]}')
    if 'Connection' in out or 'timeout' in out.lower() or 'Error' in out:
        print('HF Mirror failed, trying direct...')
        out2, _ = run_cmd(client, f'python3 -c "from huggingface_hub import HfApi; api=HfApi(); print(api.model_info(\"openai/clip-vit-large-patch14\", timeout=15).modelId)" 2>&1', timeout=30)
        print(f'Direct: {out2[:300]}')
        if 'Connection' in out2 or 'timeout' in out2.lower():
            print('HF is NOT accessible. Will use local model cache.')
        else:
            print('HF direct is accessible.')

    # Step 3: Install required packages
    print('\n=== Step 3: Install Dependencies ===')
    pkgs = 'torchvision pandas tqdm pillow'
    out, err = run_cmd(client, f'pip install {pkgs} -q 2>&1 | tail -3', timeout=120)
    print(out[:500])

    # Step 4: Run inference
    print('\n=== Step 4: Run Inference ===')
    print(f'Starting inference at {time.strftime("%H:%M:%S")}...')
    out, err = run_cmd(client, f'cd {SERVER_PROJECT} && python3 server_zero_shot.py 2>&1', timeout=1800)
    print(out[-3000:] if len(out) > 3000 else out)
    if err:
        print(f'STDERR: {err[-1000:]}')

    # Step 5: Download results
    print('\n=== Step 5: Download Results ===')
    remote_output = f'{SERVER_PROJECT}/meteor_zs_output'
    local_output = os.path.join(LOCAL_DIR, 'meteor_zs_output')
    os.makedirs(local_output, exist_ok=True)

    # List remote files
    out, _ = run_cmd(client, f'ls {remote_output}/')
    for fname in out.split('\n'):
        fname = fname.strip()
        if fname and fname.endswith('.csv'):
            try:
                download_file(client, f'{remote_output}/{fname}',
                            os.path.join(local_output, fname))
            except Exception as e:
                print(f'  Failed to download {fname}: {e}')

    client.close()
    print('\nDone!')

if __name__ == '__main__':
    main()
