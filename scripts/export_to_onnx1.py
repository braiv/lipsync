import torch
from omegaconf import OmegaConf
import sys
import time
from pathlib import Path

# 🛠️ Add LatentSync source path so imports work
sys.path.append("/home/cody_braiv_co/latent-sync")

# ✅ Import the UNet3DConditionModel
from latentsync.models.unet import UNet3DConditionModel

# 📂 Define all relevant file paths
config_path = "/home/cody_braiv_co/latent-sync/configs/unet/stage2.yaml"
ckpt_path = "/home/cody_braiv_co/latent-sync/checkpoints/latentsync_unet.pt"
onnx_path = "/home/cody_braiv_co/braiv-lipsync/scripts/latentsync_minimal.onnx"

# 📄 Load YAML config
print("📄 Loading config...")
config = OmegaConf.load(config_path)
print("✅ Config loaded.")

# 🧠 Initialize the model from config
print("🧠 Initializing model...")
model = UNet3DConditionModel(**config.model)
print("✅ Model initialized.")

# 📦 Load checkpoint and process state_dict
print("📦 Loading checkpoint...")
checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint.get("state_dict", checkpoint)
state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
print("✅ Checkpoint loaded and model ready.")

# 🧪 Prepare dummy inputs (minimized size for fast test export)
print("🔧 Creating dummy input...")
sample_input = torch.randn(1, 13, 2, 32, 32)  # (B, C, T, H, W)
timesteps = torch.tensor([10])  # ⏱️ Provide dummy timestep
print("✅ Dummy input created.")

# ⏱️ Export the model to ONNX
print(f"📤 Exporting to ONNX: {onnx_path}")
start = time.time()
with torch.no_grad():
    torch.onnx.export(
        model,
        (sample_input, timesteps),
        onnx_path,
        input_names=["sample", "timesteps"],
        output_names=["output"],
        dynamic_axes={
            "sample": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
    )
print(f"✅ Export complete: {onnx_path}")
print(f"⏱️ Time taken: {round(time.time() - start, 2)} seconds")
