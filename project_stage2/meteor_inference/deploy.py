"""Deploy Meteor-Mamba and run inference on the server"""
import paramiko
import sys
import os
import time

HOST = os.environ["METEORITE_SSH_HOST"]
PORT = int(os.environ.get("METEORITE_SSH_PORT", "22"))
USER = os.environ["METEORITE_SSH_USER"]
PASSWORD = os.environ["METEORITE_SSH_PASSWORD"]

SERVER_PROJECT = '/data/数据科学实践project'
SERVER_METEOR = '/tmp/Meteor'
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
    print('Connected to server.')
    return client


def run_cmd(client, cmd, desc='', timeout=600):
    """Run command on server and print output in real-time."""
    if desc:
        print(f'\n{"="*60}')
        print(f'>>> {desc}')
        print(f'{"="*60}')

    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out_chunks = []
    for line in iter(stdout.readline, ''):
        if line:
            print(line.rstrip())
            out_chunks.append(line)
    err = stderr.read().decode().strip()
    if err:
        print(f'[STDERR]: {err[:500]}')
    return ''.join(out_chunks), err


def upload_file(client, local_path, remote_path):
    """Upload a single file to the server."""
    sftp = client.open_sftp()
    try:
        sftp.put(local_path, remote_path)
        print(f'  Uploaded: {os.path.basename(local_path)} -> {remote_path}')
    finally:
        sftp.close()


def upload_dir(client, local_dir, remote_dir):
    """Upload all files in a directory to the server."""
    sftp = client.open_sftp()
    try:
        # Create remote dir
        try:
            sftp.mkdir(remote_dir)
        except IOError:
            pass  # Already exists

        for fname in os.listdir(local_dir):
            local_path = os.path.join(local_dir, fname)
            remote_path = f'{remote_dir}/{fname}'
            if os.path.isfile(local_path):
                sftp.put(local_path, remote_path)
                print(f'  Uploaded: {fname}')
    finally:
        sftp.close()


def main():
    client = connect()

    # ==========================================
    # Step 1: Install base dependencies
    # ==========================================
    run_cmd(client,
        'pip install transformers accelerate bitsandbytes --timeout 120',
        'Installing transformers, accelerate, bitsandbytes')

    run_cmd(client,
        'pip install pandas tqdm pillow --timeout 60',
        'Installing data processing packages')

    # ==========================================
    # Step 2: Install mamba-ssm and causal-conv1d
    # ==========================================
    run_cmd(client,
        'pip install causal-conv1d --no-build-isolation 2>&1',
        'Installing causal-conv1d')

    run_cmd(client,
        'pip install mamba-ssm --no-build-isolation 2>&1',
        'Installing mamba-ssm')

    # ==========================================
    # Step 3: Clone Meteor repository
    # ==========================================
    run_cmd(client,
        f'if [ -d "{SERVER_METEOR}" ]; then rm -rf {SERVER_METEOR}; fi; '
        f'git clone https://github.com/ByungKwanLee/Meteor.git {SERVER_METEOR} 2>&1',
        'Cloning Meteor repository')

    # ==========================================
    # Step 4: Install Meteor
    # ==========================================
    run_cmd(client,
        f'cd {SERVER_METEOR} && pip install -e . 2>&1',
        'Installing Meteor package')

    # ==========================================
    # Step 5: Verify installations
    # ==========================================
    verify_out, _ = run_cmd(client,
        'python -c "'
        'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\"); '
        'import mamba_ssm; print(\"mamba_ssm: OK\"); '
        'import causal_conv1d; print(\"causal_conv1d: OK\"); '
        'import transformers; print(f\"transformers: {transformers.__version__}\"); '
        'import accelerate; print(f\"accelerate: {accelerate.__version__}\"); '
        'import bitsandbytes; print(\"bitsandbytes: OK\"); '
        'print(\"GPU:\", torch.cuda.get_device_name(0)); '
        'print(f\"VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB\")"',
        'Verifying installations')

    print('\n' + '='*60)
    print('>>> DEPLOYMENT COMPLETE')
    print('='*60)

    client.close()


if __name__ == '__main__':
    main()
