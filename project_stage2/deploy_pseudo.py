"""Deploy and run pseudo-label self-training on GPU server"""
import paramiko
import os
import time
import sys

HOST = os.environ["METEORITE_SSH_HOST"]
PORT = int(os.environ.get("METEORITE_SSH_PORT", "22"))
USER = os.environ["METEORITE_SSH_USER"]
PASSWORD = os.environ["METEORITE_SSH_PASSWORD"]
SERVER_PROJECT = '/data/数据科学实践project'
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("Deploy the self-training experiment")
print("=" * 60)

# 1. Connect
print("\n[1/5] Connect to the server...")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
    print(f"  Connected to {HOST}:{PORT}")
except Exception as e:
    print(f"  Connection failed: {e}")
    sys.exit(1)

# 2. Upload files
print("\n[2/5] Upload files...")
sftp = client.open_sftp()

files_to_upload = [
    'train_dinov2_pseudo.py',
    'submission_dinov2_L14_top85.csv',  # pseudo-label source
]

for fname in files_to_upload:
    local_path = os.path.join(LOCAL_DIR, fname)
    remote_path = f'{SERVER_PROJECT}/{fname}'
    if os.path.exists(local_path):
        sftp.put(local_path, remote_path)
        print(f"  {fname} -> {remote_path}")
    else:
        print(f"  [WARNING] {fname} missing locally; skipping upload")

sftp.close()

# 3. Check GPU and dependencies
print("\n[3/5] Check environment...")
stdin, stdout, stderr = client.exec_command(
    'nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>&1 && '
    'python3 -c "import torch; print(f\'PyTorch: {torch.__version__}\'); '
    'print(f\'CUDA available: {torch.cuda.is_available()}\'); '
    'print(f\'GPU: {torch.cuda.get_device_name(0)}\')" 2>&1',
    timeout=30
)
out = stdout.read().decode().strip()
err = stderr.read().decode().strip()
print("  " + out.replace('\n', '\n  '))
if err:
    print(f"  STDERR: {err[:500]}")

# 4. Verify pseudo-label file on server
print("\n[4/5] Verify remote files...")
stdin, stdout, stderr = client.exec_command(
    f'ls -la {SERVER_PROJECT}/submission_dinov2_L14_top85.csv 2>&1 && '
    f'head -3 {SERVER_PROJECT}/submission_dinov2_L14_top85.csv && echo "---" && '
    f'wc -l {SERVER_PROJECT}/submission_dinov2_L14_top85.csv',
    timeout=15
)
print("  " + stdout.read().decode().strip().replace('\n', '\n  '))

# 5. Run training
print(f"\n[5/5] Start self-training ({time.strftime('%H:%M:%S')})...")
print("=" * 60)

stdin, stdout, stderr = client.exec_command(
    f'cd {SERVER_PROJECT} && python3 train_dinov2_pseudo.py 2>&1',
    timeout=7200  # 2 hours
)

# Stream output
all_output = []
for line in iter(stdout.readline, ''):
    if line:
        print(line.rstrip())
        all_output.append(line)

err_out = stderr.read().decode().strip()
if err_out:
    print(f'\nSTDERR:\n{err_out[-2000:]}')

print("\n" + "=" * 60)
print("Training complete!")

# Download results
print("\n Download results...")
sftp = client.open_sftp()

# Find latest pseudo output dir
stdin2, stdout2, _ = client.exec_command(
    f'ls -dt {SERVER_PROJECT}/runs/dinov2_pseudo_*/ 2>/dev/null | head -1'
)
latest_dir = stdout2.read().decode().strip()
print(f"  Latest run: {latest_dir}")

if latest_dir:
    local_output = os.path.join(LOCAL_DIR, os.path.basename(latest_dir.rstrip('/')))
    os.makedirs(local_output, exist_ok=True)

    stdin3, stdout3, _ = client.exec_command(f'ls {latest_dir}')
    for fname in stdout3.read().decode().strip().split('\n'):
        fname = fname.strip()
        if fname:
            remote_path = f'{latest_dir}/{fname}'
            local_path = os.path.join(local_output, fname)
            try:
                sftp.get(remote_path, local_path)
                print(f"  {fname}")
            except Exception as e:
                print(f"  Failed: {fname}: {e}")

sftp.close()
client.close()
print("\n Finished!")
