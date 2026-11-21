"""Install causal-conv1d in the background along with mamba-ssm"""
import os
import paramiko
import time
import sys

HOST = os.environ["METEORITE_SSH_HOST"]
PORT = int(os.environ.get("METEORITE_SSH_PORT", "22"))
USER = os.environ["METEORITE_SSH_USER"]
PASSWORD = os.environ["METEORITE_SSH_PASSWORD"]

def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
    return client

def run_cmd(client, cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    return stdout.read().decode().strip(), stderr.read().decode().strip()

# Step 1: Start causal-conv1d installation in background
print("=== Starting causal-conv1d installation (background) ===")
client = connect()
cmd = (
    'export PATH=/usr/local/cuda-12.8/bin:$PATH && '
    'export CUDA_HOME=/usr/local/cuda-12.8 && '
    'nohup pip install --break-system-packages causal-conv1d --no-build-isolation '
    '> /tmp/causal_install.log 2>&1 &'
    'echo $!'
)
stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
pid = stdout.read().decode().strip()
print(f'Install PID: {pid}')
client.close()

# Step 2: Poll until complete
print('Waiting for compilation (this may take 5-10 minutes)...')
for i in range(40):  # Max 20 minutes
    time.sleep(30)
    client = connect()
    out, err = run_cmd(client, 'tail -5 /tmp/causal_install.log 2>/dev/null')
    client.close()

    if out:
        print(f'[{i*30}s] {out[-200:]}')

    # Check if pip process still running
    client = connect()
    ps_out, _ = run_cmd(client, f'ps aux | grep "pip install.*causal" | grep -v grep | wc -l')
    client.close()
    if ps_out.strip() == '0':
        print("Installation process finished!")
        break
else:
    print("Timeout waiting for installation")

# Step 3: Check result
print("\n=== Checking installation result ===")
client = connect()
out, err = run_cmd(client, 'cat /tmp/causal_install.log')
client.close()
print(out[-1000:] if len(out) > 1000 else out)
if err:
    print("STDERR:", err[:500])

# Step 4: Verify imports
print("\n=== Verifying causal-conv1d ===")
client = connect()
out, err = run_cmd(client, 'python -c "import causal_conv1d; print(\'causal_conv1d OK\')" 2>&1')
client.close()
print(out if out else err[:500])

# Step 5: Install mamba-ssm if causal-conv1d succeeded
if 'OK' in out:
    print("\n=== Installing mamba-ssm ===")
    client = connect()
    cmd = (
        'export PATH=/usr/local/cuda-12.8/bin:$PATH && '
        'export CUDA_HOME=/usr/local/cuda-12.8 && '
        'nohup pip install --break-system-packages mamba-ssm --no-build-isolation '
        '> /tmp/mamba_install.log 2>&1 &'
        'echo $!'
    )
    stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
    pid2 = stdout.read().decode().strip()
    print(f'Install PID: {pid2}')
    client.close()

    for i in range(40):
        time.sleep(30)
        client = connect()
        out, _ = run_cmd(client, 'tail -5 /tmp/mamba_install.log 2>/dev/null')
        client.close()
        if out:
            print(f'[{i*30}s] {out[-200:]}')
        client = connect()
        ps_out, _ = run_cmd(client, 'ps aux | grep "pip install.*mamba" | grep -v grep | wc -l')
        client.close()
        if ps_out.strip() == '0':
            print("Installation process finished!")
            break

    client = connect()
    out, _ = run_cmd(client, 'cat /tmp/mamba_install.log')
    client.close()
    print(out[-1000:] if len(out) > 1000 else out)

    print("\n=== Verifying mamba-ssm ===")
    client = connect()
    out, _ = run_cmd(client, 'python -c "import mamba_ssm; print(\'mamba_ssm OK\')" 2>&1')
    client.close()
    print(out)
else:
    print("\nCannot proceed with mamba-ssm - causal-conv1d failed")

print("\n=== DONE ===")
