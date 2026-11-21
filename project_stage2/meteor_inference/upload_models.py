"""Upload Meteor models from local to server HF cache"""
import paramiko
import os
import sys
import time

HOST = os.environ["METEORITE_SSH_HOST"]
PORT = int(os.environ.get("METEORITE_SSH_PORT", "22"))
USER = os.environ["METEORITE_SSH_USER"]
PASSWORD = os.environ["METEORITE_SSH_PASSWORD"]

LOCAL_MODELS = r'H:\数据科学与大数据技术\数据科学实践\project1\project_stage2\meteor_models'
REMOTE_CACHE = '/root/.cache/huggingface/hub'


def upload_dir(sftp, local_dir, remote_dir, desc=''):
    total_size = 0
    total_files = sum(1 for _, _, fs in os.walk(local_dir) for f in fs)
    uploaded = 0
    start = time.time()

    for root, dirs, files in os.walk(local_dir):
        rel_path = os.path.relpath(root, local_dir)
        remote_path = os.path.join(remote_dir, rel_path).replace('\\', '/')
        try:
            sftp.mkdir(remote_path)
        except IOError:
            pass
        for f in files:
            local_f = os.path.join(root, f)
            remote_f = os.path.join(remote_path, f).replace('\\', '/')
            size_mb = os.path.getsize(local_f) / 1e6
            total_size += size_mb
            print(f'  [{desc}] {size_mb:.1f}MB - {f[:60]}', end='', flush=True)
            sftp.put(local_f, remote_f)
            uploaded += 1
            elapsed = time.time() - start
            speed = total_size / elapsed if elapsed > 0 else 0
            print(f' OK ({uploaded}/{total_files}) [{speed:.1f} MB/s]')

    elapsed = time.time() - start
    print(f'  [{desc}] Done: {total_size:.0f}MB in {elapsed:.0f}s ({total_size/elapsed:.1f} MB/s)')


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15)
    sftp = c.open_sftp()

    # Step 1: Upload Meteor-Mamba
    local_mmamba = os.path.join(LOCAL_MODELS, 'meteor_mamba')
    remote_mmamba = os.path.join(REMOTE_CACHE, 'models--BK-Lee--Meteor-Mamba')
    print('=== Uploading Meteor-Mamba ===')
    upload_dir(sftp, local_mmamba, remote_mmamba, 'Mamba')

    # Step 2: Upload Meteor-MLM
    local_mlm = os.path.join(LOCAL_MODELS, 'meteor_mlm')
    remote_mlm = os.path.join(REMOTE_CACHE, 'models--BK-Lee--Meteor-MLM')
    print('\n=== Uploading Meteor-MLM (17GB, this will take a while) ===')
    upload_dir(sftp, local_mlm, remote_mlm, 'MLM')

    sftp.close()
    c.close()
    print('\n=== ALL UPLOADS COMPLETE ===')


if __name__ == '__main__':
    main()
