"""Check server status"""
import paramiko
import os

HOST = os.environ["METEORITE_SSH_HOST"]
PORT = int(os.environ.get("METEORITE_SSH_PORT", "22"))
USER = os.environ["METEORITE_SSH_USER"]
PASSWORD = os.environ["METEORITE_SSH_PASSWORD"]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)

# Run commands
cmd = """
echo "=== Process ==="
ps aux | grep server_dinov2 | grep -v grep
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "=== Output dir ==="
ls -la /data/数据科学实践project/dinov2_natural_output/
echo "=== Any CSV outputs ==="
find /data/数据科学实践project -name "*.csv" -newer /data/数据科学实践project/server_dinov2_classify.py 2>/dev/null | head -10
echo "=== Last lines of script ==="
tail -1 /data/数据科学实践project/server_dinov2_classify.py
"""
stdin, stdout, stderr = client.exec_command(cmd, timeout=15)
print(stdout.read().decode().strip())
if stderr:
    err = stderr.read().decode().strip()
    if err:
        print("STDERR:", err[:500])

client.close()
