"""检查学校服务器环境"""
import os
import paramiko
import sys

host = os.environ["METEORITE_SSH_HOST"]
port = int(os.environ.get("METEORITE_SSH_PORT", "22"))
user = os.environ["METEORITE_SSH_USER"]
password = os.environ["METEORITE_SSH_PASSWORD"]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    client.connect(host, port=port, username=user, password=password, timeout=15)
    print('=== Successfully Connected ===')

    commands = {
        'GPU': 'nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>&1',
        'Python': 'python3 --version 2>&1; python --version 2>&1; which python3; which python',
        'CUDA': 'nvcc --version 2>&1 | tail -1; nvidia-smi 2>&1 | head -5',
        'Disk': 'df -h / 2>&1; echo "---"; ls /data 2>/dev/null; ls /home 2>/dev/null',
        'Conda': 'which conda 2>&1; conda info --envs 2>&1 | head -10',
        'Pip packages': 'pip3 list 2>/dev/null | grep -iE "torch|transformers|mamba|bitsandbytes|causal" || pip list 2>/dev/null | grep -iE "torch|transformers|mamba|bitsandbytes|causal"',
        'OS': 'cat /etc/os-release 2>/dev/null | head -4; uname -a',
        'RAM': 'free -h 2>/dev/null | head -2',
    }

    for name, cmd in commands.items():
        stdin, stdout, stderr = client.exec_command(cmd)
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()
        print(f'\n=== {name} ===')
        if output:
            print(output)
        if error:
            print(f'[stderr]: {error}')

    client.close()

except Exception as e:
    print(f'Connection failed: {e}')
    sys.exit(1)
