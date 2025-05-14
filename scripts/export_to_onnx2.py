import torch
from omegaconf import OmegaConf
import sys
import time
from pathlib import Path
import os

# 🛠️ Add LatentSync source path so imports work
sys.path.append("/home/cody_braiv_co/latent-sync")

# ✅ Import the UNet3DConditionModel
from latentsync.models.unet import UNet3DConditionModel

# 📂 Define all relevant file paths
config_path = "/home/cody_braiv_co/latent-sync/configs/unet/stage2.yaml"
ckpt_path = "/home/cody_braiv_co/latent-sync/checkpoints/latentsync_unet.pt"
onnx_path = "/home/cody_braiv_co/braiv-lipsync/scripts/latentsync_unet.onnx"

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
sample_input = torch.randn(1, 13, 8, 64, 64)  # (B, C, T, H, W)
timesteps = torch.tensor([10])
print("✅ Dummy input created.")

# ✅ Optional: Test forward pass before export
print("🧪 Testing model forward pass...")
try:
    with torch.no_grad():
        _ = model(sample_input, timesteps)
    print("✅ Forward pass successful.")
except Exception as e:
    print("❌ Forward pass failed:", e)
    exit(1)

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
        "sample": {
            0: "batch_size",     # Dynamic
            2: "num_frames",     # Dynamic
            # 3: "height",       # Optional — only if you test it
            # 4: "width"         # Optional — only if you test it
        },
        "output": {
            0: "batch_size",
            2: "num_frames",
            # 3: "height",
            # 4: "width"
        }
    },
    opset_version=17,
    )
print(f"✅ Export complete: {onnx_path}")
print(f"⏱️ Time taken: {round(time.time() - start, 2)} seconds")

# 📁 Move relevant files to new folder after export
DEST_FOLDER = "onnx_weights"
print(f"📁 Moving files to ./{DEST_FOLDER}...")

os.makedirs(DEST_FOLDER, exist_ok=True)
os.system(f'mv *.weight {DEST_FOLDER}/ 2>/dev/null')
os.system(f'mv *.bias {DEST_FOLDER}/ 2>/dev/null')
os.system(f'mv *.pe {DEST_FOLDER}/ 2>/dev/null')
os.system(f'mv onnx__* {DEST_FOLDER}/ 2>/dev/null')

print(f"✅ All .weight, .bias, .pe, and onnx__* files moved to {DEST_FOLDER}")
