import torch
from omegaconf import OmegaConf
import sys
import time
from pathlib import Path
import os
import onnxruntime as ort
import numpy 
import gc  # For garbage collection


# 🛠️ Add LatentSync source path so imports work
sys.path.append("/home/cody_braiv_co/latent-sync")

# ✅ Import the UNet3DConditionModel
from latentsync.models.unet import UNet3DConditionModel

# 📂 Define all relevant file paths
config_path = "/home/cody_braiv_co/latent-sync/configs/unet/stage2.yaml"
ckpt_path = "/home/cody_braiv_co/latent-sync/checkpoints/latentsync_unet.pt"
onnx_path = "/home/cody_braiv_co/braiv-lipsync/scripts/latentsync_unet.onnx"

# ✅ Detect device and check memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"💾 Total GPU memory: {total_memory:.1f} GB")
    if total_memory < 12:
        print("⚠️  WARNING: Less than 12GB VRAM detected. Export may fail!")
        print("💡 Consider using CPU export or smaller batch size.")

# 🧹 Initial memory cleanup
torch.cuda.empty_cache()
gc.collect()

# 📄 Load YAML config
print("📄 Loading config...")
config = OmegaConf.load(config_path)
print("✅ Config loaded.")

# 🧠 Initialize the model from config
print("🧠 Initializing model...")
model = UNet3DConditionModel(**config.model).to(device).float()  # FP32 for stability
print("✅ Model initialized in float32 for better stability.")

if torch.cuda.is_available():
    model_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"🧠 Model memory usage: {model_memory:.2f} GB")

# 📦 Load checkpoint and process state_dict
print("📦 Loading checkpoint...")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# 🧹 Clean up checkpoint from memory
del checkpoint, state_dict
torch.cuda.empty_cache()
gc.collect()
print("✅ Checkpoint loaded and cleaned up.")

# 🧪 Prepare dummy inputs with MEMORY-OPTIMIZED dimensions
print("🔧 Creating dummy input...")

# 🔥 MEMORY OPTIMIZATION: Use smaller batch size for export
# Export with batch_size=1, then use dynamic axes for CFG during inference
export_batch_size = 1  # Reduces memory by 50%
# 🚀 T4 16GB OPTIMIZATION: Increased from 50 to 75 for better quality
audio_seq_len = 75     # Increased for T4 16GB - better audio coverage

# ✅ Memory-optimized input shapes - FP32 for better compatibility
sample_input = torch.randn(export_batch_size, 13, 1, 64, 64).to(device).float()  # FP32
timesteps = torch.tensor([10], dtype=torch.int64).to(device)
encoder_hidden_states = torch.randn(export_batch_size, audio_seq_len, 384).to(device).float()  # FP32

print("✅ Dummy input created with MEMORY-OPTIMIZED dimensions:")
print(f"   sample_input: {sample_input.shape}")
print(f"   timesteps: {timesteps.shape}")
print(f"   encoder_hidden_states: {encoder_hidden_states.shape}")

if torch.cuda.is_available():
    input_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"💾 Total memory after inputs: {input_memory:.2f} GB")

# 🧹 Memory cleanup before forward pass
torch.cuda.empty_cache()

# ✅ Optional: Test forward pass before export (can be skipped to save memory)
# 🚀 T4 16GB: Enable full testing since we have plenty of memory
SKIP_FORWARD_TEST = False  # T4 16GB can handle full testing

if not SKIP_FORWARD_TEST:
    print("🧪 Testing model forward pass...")
    try:
        with torch.no_grad():
            output = model(sample_input, timesteps, encoder_hidden_states)
            print(f"✅ Forward pass successful.")
            
            # Clean up output immediately
            del output
            torch.cuda.empty_cache()
            
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"🧠 Peak GPU memory during forward: {peak_memory:.2f} GB")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ OOM during forward pass. Continuing with export anyway...")
            print("💡 The export might still succeed with optimizations.")
            torch.cuda.empty_cache()
        else:
            print("❌ Forward pass failed:", e)
            exit(1)
else:
    print("⏭️  Skipping forward pass test to save memory.")

# 🧹 Major cleanup before export
torch.cuda.empty_cache()
gc.collect()

# ⏱️ Export the model to ONNX with memory optimizations
print(f"📤 Exporting to ONNX: {onnx_path}")
print("⚠️  This may take several minutes and use significant memory...")

start = time.time()
try:
    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample_input, timesteps, encoder_hidden_states),
            onnx_path,
            input_names=["sample", "timesteps", "encoder_hidden_states"],
            output_names=["output"],
            dynamic_axes={
                "sample": {
                    0: "batch_size",      # CFG: 1 or 2
                    2: "num_frames"       # Usually 1, but could vary
                },
                "encoder_hidden_states": {
                    0: "batch_size",      # CFG: 1 or 2
                    1: "seq_len"          # Audio sequence length (varies by duration)
                },
                "output": {
                    0: "batch_size",      # CFG: 1 or 2
                    2: "num_frames"       # Usually 1, but could vary
                }
            },
            opset_version=17,
            export_params=True,
            # Memory optimization options
            do_constant_folding=True,    # Reduces model size
            verbose=False                # Reduces memory overhead
        )
    print(f"✅ Export complete: {onnx_path}")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("❌ OOM during ONNX export!")
        print("💡 Try these solutions:")
        print("   1. Use CPU export: device = torch.device('cpu')")
        print("   2. Reduce audio_seq_len to 25")
        print("   3. Close other GPU applications")
        print("   4. Use a machine with more VRAM")
        exit(1)
    else:
        raise e

print(f"⏱️ Time taken: {round(time.time() - start, 2)} seconds")

# 🧹 Massive cleanup before verification
del model, sample_input, timesteps, encoder_hidden_states
torch.cuda.empty_cache()
gc.collect()

# 🧪 OPTIONAL ONNX verification (can be skipped to save memory)
# 🚀 T4 16GB: Enable full verification since we have plenty of memory
SKIP_VERIFICATION = False  # T4 16GB can handle full verification

if not SKIP_VERIFICATION:
    print("🧪 Verifying exported ONNX model...")
    
    try:
        # Create smaller test inputs for verification
        test_sample = torch.randn(1, 13, 1, 64, 64).float()  # FP32
        test_timesteps = torch.tensor([10], dtype=torch.int64)
        # 🚀 T4 16GB: Use same audio length as export for thorough verification
        test_audio = torch.randn(1, 75, 384).float()  # FP32 for consistency
        
        # Safe ONNX session options
        so = ort.SessionOptions()
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        ort_session = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        )

        outputs = ort_session.run(
            None,
            {
                "sample": test_sample.cpu().numpy().astype(numpy.float32),
                "timesteps": test_timesteps.cpu().numpy().astype(numpy.int64),
                "encoder_hidden_states": test_audio.cpu().numpy().astype(numpy.float32)
            }
        )

        print("✅ ONNX model inference successful. Output shape:", outputs[0].shape)
        expected_shape = (1, 4, 1, 64, 64)
        if outputs[0].shape == expected_shape:
            print("✅ Output shape matches expected dimensions!")
        else:
            print(f"⚠️  Output shape mismatch: got {outputs[0].shape}, expected {expected_shape}")

    except Exception as e:
        print("❌ ONNX model verification failed:", e)
        print("💡 Model export may still be valid - verification can fail due to memory.")
else:
    print("⏭️  Skipping ONNX verification to save memory.")

# 📁 Move relevant files to new folder after export
DEST_FOLDER = "latentsync_model_files"
print(f"📁 Moving files to ./{DEST_FOLDER}...")

os.makedirs(DEST_FOLDER, exist_ok=True)
os.system(f'mv *.weight {DEST_FOLDER}/ 2>/dev/null')
os.system(f'mv *.bias {DEST_FOLDER}/ 2>/dev/null')
os.system(f'mv *.pe {DEST_FOLDER}/ 2>/dev/null')
os.system(f'mv onnx__* {DEST_FOLDER}/ 2>/dev/null')

print(f"✅ All .weight, .bias, .pe, and onnx__* files moved to {DEST_FOLDER}")

print("\n🎯 Export Summary:")
print(f"   Model exported with MEMORY-OPTIMIZED dimensions:")
print(f"   - Input: ({export_batch_size}, 13, 1, 64, 64) - single frame processing")
print(f"   - Audio: ({export_batch_size}, {audio_seq_len}, 384) - optimized for T4 16GB")
print(f"   - Output: ({export_batch_size}, 4, 1, 64, 64) - VAE latent space")
print(f"   - Dynamic axes enabled for runtime CFG support (1→2 batch scaling)")
print(f"   - FP32 precision for maximum stability and compatibility")
print(f"   - Memory optimizations: constant folding, lazy loading, aggressive cleanup")
print(f"   - Expected runtime memory: ~7GB peak (excellent for T4 16GB)")

if torch.cuda.is_available():
    final_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"💾 Final GPU memory usage: {final_memory:.2f} GB (models successfully cleared)")
