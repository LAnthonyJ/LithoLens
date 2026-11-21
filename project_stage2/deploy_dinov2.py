"""Deploy and run DINOv2 classifier on GPU server"""
import paramiko
import os
import time

HOST = os.environ["METEORITE_SSH_HOST"]
PORT = int(os.environ.get("METEORITE_SSH_PORT", "22"))
USER = os.environ["METEORITE_SSH_USER"]
PASSWORD = os.environ["METEORITE_SSH_PASSWORD"]
SERVER_PROJECT = '/data/数据科学实践project'
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
print('Connected.')

# Upload script
sftp = client.open_sftp()
local_script = os.path.join(LOCAL_DIR, 'server_dinov2_classify.py')
remote_script = f'{SERVER_PROJECT}/server_dinov2_classify.py'
sftp.put(local_script, remote_script)
sftp.close()
print(f'Uploaded script.')

# Install missing packages
print('\nInstalling xgboost, lightgbm...')
stdin, stdout, stderr = client.exec_command(
    'pip install xgboost lightgbm scikit-learn --break-system-packages -q 2>&1 | tail -5',
    timeout=120
)
out = stdout.read().decode().strip()
err = stderr.read().decode().strip()
print(out)
if err:
    print(f'STDERR: {err[:500]}')

# Run inference
print(f'\nRunning DINOv2 classifier at {time.strftime("%H:%M:%S")}...')
stdin, stdout, stderr = client.exec_command(
    f'cd {SERVER_PROJECT} && python3 server_dinov2_classify.py 2>&1',
    timeout=3600
)

# Stream output in real-time
all_output = []
for line in iter(stdout.readline, ''):
    if line:
        print(line.rstrip())
        all_output.append(line)

err_out = stderr.read().decode().strip()
if err_out:
    print(f'\nSTDERR:\n{err_out[-2000:]}')

# Download results
print('\nDownloading results...')
sftp = client.open_sftp()
remote_output = f'{SERVER_PROJECT}/dinov2_natural_output'
local_output = os.path.join(LOCAL_DIR, 'dinov2_natural_output')
os.makedirs(local_output, exist_ok=True)

stdin2, stdout2, _ = client.exec_command(f'ls {remote_output}/')
for fname in stdout2.read().decode().strip().split('\n'):
    fname = fname.strip()
    if fname:
        remote_path = f'{remote_output}/{fname}'
        local_path = os.path.join(local_output, fname)
        try:
            sftp.get(remote_path, local_path)
            print(f'  Downloaded: {fname}')
        except Exception as e:
            print(f'  Failed: {fname}: {e}')

sftp.close()
client.close()
print('\nDone!')
