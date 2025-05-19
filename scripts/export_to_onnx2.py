import torch
from omegaconf import OmegaConf
import sys
import time
from pathlib import Path
import os
import onnxruntime as ort
import numpy as np


# 🛠️ Add LatentSync source path so imports work
sys.path.append("/home/cody_braiv_co/latent-sync")

# ✅ Import the UNet3DConditionModel
from latentsync.models.unet import UNet3DConditionModel

# 📂 Define all relevant file paths
config_path = "/home/cody_braiv_co/latent-sync/configs/unet/stage2.yaml"
ckpt_path = "/home/cody_braiv_co/latent-sync/checkpoints/latentsync_unet.pt"
onnx_path = "/home/cody_braiv_co/braiv-lipsync/scripts/latentsync_unet.onnx"

# ✅ Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# 📄 Load YAML config
print("📄 Loading config...")
config = OmegaConf.load(config_path)
print("✅ Config loaded.")

# 🧠 Initialize the model from config
print("🧠 Initializing model...")
model = UNet3DConditionModel(**config.model).to(device)
print("✅ Model initialized.")

# 📦 Load checkpoint and process state_dict
print("📦 Loading checkpoint...")
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint.get("state_dict", checkpoint)
state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
print("✅ Checkpoint loaded and model ready.")

# 🧪 Prepare dummy inputs
print("🔧 Creating dummy input...")
sample_input = torch.randn(1, 13, 4, 32, 32).to(device)  # (B, C, T, H, W)
timesteps = torch.tensor([10.0], dtype=torch.float32).to(device)  # Match ONNX float32

encoder_hidden_states = torch.randn(1, 16384, 384).to(device)

print("✅ Dummy input created. encoder_hidden_states:", encoder_hidden_states.shape)

# ✅ Optional: Test forward pass before export
print("🧪 Testing model forward pass...")
try:
    with torch.no_grad():
        _ = model(sample_input, timesteps, encoder_hidden_states)
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
        (sample_input, timesteps, encoder_hidden_states),
        onnx_path,
        input_names=["sample", "timesteps", "encoder_hidden_states"],
        output_names=["output"],
        dynamic_axes={
            "sample": {
                0: "batch_size",
                2: "num_frames",
                3: "height",       # Optional — only if you test it
                4: "width"         # Optional — only if you test it
            },
             "encoder_hidden_states": {
                 0: "batch_size", 
                 1: "seq_len"
            },
            "output": {
                0: "batch_size",
                2: "num_frames",
                3: "height",
                4: "width"
            }
        },
        opset_version=17,
        export_params=True # This includes the weights!
    )
print(f"✅ Export complete: {onnx_path}")
print(f"⏱️ Time taken: {round(time.time() - start, 2)} seconds")

# 🧪 ONNX inference check
print("🧪 Verifying exported ONNX model with GPU...")

sample_input = torch.randn(1, 13, 2, 32, 32).to(device)
encoder_hidden_states = torch.randn(1, 2048, 384).to(device)  # much smaller seq_len

try:
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
    )

    outputs = ort_session.run(
        None,
        {
            "sample": sample_input.cpu().numpy(),  # ONNXRuntime expects NumPy arrays on CPU
            "timesteps": timesteps.cpu().numpy(),
            "encoder_hidden_states": encoder_hidden_states.cpu().numpy()
        }
    )

    print("✅ ONNX model inference successful. Output shape:", outputs[0].shape)

except Exception as e:
    print("❌ ONNX model verification failed:", e)


# 📁 Move relevant files to new folder after export
DEST_FOLDER = "latentsync_model_files"
print(f"📁 Moving files to ./{DEST_FOLDER}...")

os.makedirs(DEST_FOLDER, exist_ok=True)
os.system(f'mv *.weight {DEST_FOLDER}/ 2>/dev/null')
os.system(f'mv *.bias {DEST_FOLDER}/ 2>/dev/null')
os.system(f'mv *.pe {DEST_FOLDER}/ 2>/dev/null')
os.system(f'mv onnx__* {DEST_FOLDER}/ 2>/dev/null')

print(f"✅ All .weight, .bias, .pe, and onnx__* files moved to {DEST_FOLDER}")
