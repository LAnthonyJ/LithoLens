"""Upload CLIP model to server"""
import paramiko
import os
import time

HOST = os.environ["METEORITE_SSH_HOST"]
PORT = int(os.environ.get("METEORITE_SSH_PORT", "22"))
USER = os.environ["METEORITE_SSH_USER"]
PASSWORD = os.environ["METEORITE_SSH_PASSWORD"]

LOCAL_CLIP = r'H:\数据科学与大数据技术\数据科学实践\project1\project_stage2\meteor_models\clip-vit-large'
REMOTE_CLIP = '/data/models/clip-vit-large-patch14-336'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
sftp = c.open_sftp()

# Create remote dirs
dirs_to_create = set()
for root, dirs, files in os.walk(LOCAL_CLIP):
    rel = os.path.relpath(root, LOCAL_CLIP)
    remote_dir = os.path.join(REMOTE_CLIP, rel).replace('\\', '/')
    dirs_to_create.add(remote_dir)

for d in sorted(dirs_to_create):
    try:
        sftp.mkdir(d)
    except IOError:
        pass

# Upload files
total = 0
start = time.time()
uploaded = 0
for root, dirs, files in os.walk(LOCAL_CLIP):
    for f in files:
        local_f = os.path.join(root, f)
        size_mb = os.path.getsize(local_f) / 1e6
        rel = os.path.relpath(root, LOCAL_CLIP)
        remote_dir = os.path.join(REMOTE_CLIP, rel).replace('\\', '/')
        remote_f = (remote_dir + '/' + f).replace('\\', '/')

        if 'tf_model' in f:
            print(f'  SKIP (TF): {f} ({size_mb:.0f}MB)')
            continue

        print(f'  Uploading: {f} ({size_mb:.0f}MB)...', end='', flush=True)
        sftp.put(local_f, remote_f)
        total += size_mb
        uploaded += 1
        elapsed = time.time() - start
        print(f' OK [{total:.0f}MB in {elapsed:.0f}s, {total/elapsed:.1f} MB/s]')

sftp.close()
elapsed = time.time() - start
print(f'\nTotal: {total:.0f}MB ({uploaded} files) in {elapsed:.0f}s ({total/elapsed:.1f} MB/s)')

# Verify
print('\nServer files:')
_, out, _ = c.exec_command('ls -lh ' + REMOTE_CLIP + '/', timeout=10)
print(out.read().decode().strip())

c.close()
print('\nDone!')
