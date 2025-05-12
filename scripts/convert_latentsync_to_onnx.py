import torch
import onnx
import hashlib
from pathlib import Path
import sys

# Optional: Add the cloned LatentSync repo to PYTHONPATH if needed
sys.path.append("/home/cody_braiv_co/latent-sync")

from latentsync.models.unet import LatentSyncUNet

def calculate_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def convert_latentsync_to_onnx():
    # Define paths
    assets_dir = Path("../.assets/models")
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Load model and weights
    model = LatentSyncUNet()
    state_dict = torch.load(assets_dir / "latentsync_unet.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare dummy input
    dummy_audio = torch.randn(1, 1, 80, 16)         # (B, 1, mel, time)
    dummy_video = torch.randn(1, 3, 256, 256)       # (B, C, H, W)

    onnx_path = assets_dir / "latentsync.onnx"
    torch.onnx.export(
        model,
        (dummy_audio, dummy_video),
        str(onnx_path),
        input_names=["audio", "image"],
        output_names=["output"],
        dynamic_axes={
            "audio": {0: "batch_size"},
            "image": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=12
    )

    # Check ONNX validity
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Save hash
    hash_path = assets_dir / "latentsync.hash"
    with open(hash_path, "w") as f:
        f.write(calculate_file_hash(onnx_path))

    print(f"✅ ONNX model saved: {onnx_path}")
    print(f"🔐 Hash saved: {hash_path}")

if __name__ == "__main__":
    convert_latentsync_to_onnx()