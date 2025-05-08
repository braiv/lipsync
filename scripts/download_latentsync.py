import os
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

def calculate_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_file(url, destination):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download {url} (status code {response.status_code})")

    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def download_latentsync():
    # Define save directory
    assets_dir = Path("../.assets/models")
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Hugging Face URLs
    files = {
        "latentsync_unet.pt": "https://huggingface.co/ByteDance/LatentSync-1.5/resolve/main/latentsync_unet.pt",
        "tiny.pt": "https://huggingface.co/ByteDance/LatentSync-1.5/resolve/main/whisper/tiny.pt"
    }

    for filename, url in files.items():
        dest_path = assets_dir / filename
        print(f"⬇️ Downloading {filename} from:\n{url}")
        try:
            download_file(url, dest_path)
            print(f"✅ Saved to {dest_path}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            continue

        # Optional: Save hash
        model_hash = calculate_file_hash(dest_path)
        with open(assets_dir / f"{filename}.hash", "w") as f:
            f.write(model_hash)
        print(f"🔐 SHA256 hash saved: {filename}.hash")

if __name__ == "__main__":
    download_latentsync()