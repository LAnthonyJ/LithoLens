"""Fix CLIPVisionTower and download CLIP model"""
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 1. Patch build_module.py
build_path = '/tmp/Meteor/meteor/arch/build_module.py'
content = open(build_path, 'r').read()

# Fix load_model
old = '''    def load_model(self):
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True'''

new = '''    def load_model(self):
        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name,
            device_map=None,
            low_cpu_mem_usage=False,
            local_files_only=False,
        )
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True'''

if old in content:
    content = content.replace(old, new)
    open(build_path, 'w').write(content)
    print("Patched build_module.py")
else:
    print("Could not find old load_model - already patched?")

# 2. Try to download CLIP model
print("Downloading CLIP ViT-Large...")
from huggingface_hub import snapshot_download
try:
    path = snapshot_download(
        'openai/clip-vit-large-patch14-336',
        max_workers=1,
        local_files_only=False,
    )
    print(f"CLIP OK: {path}")
except Exception as e:
    print(f"CLIP download failed: {e}")
    print("Will try to use local files if available")
