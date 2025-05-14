import hashlib
from pathlib import Path

# ✅ Path to your ONNX model
model_path = Path("latentsync_unet.onnx")

# ✅ Output hash file with just .hash extension
hash_path = model_path.with_suffix("").with_suffix(".hash")

# ✅ Read the file and calculate SHA256
with open(model_path, "rb") as f:
    file_data = f.read()
    sha256_hash = hashlib.sha256(file_data).hexdigest()

# ✅ Write the hash to the .hash file
with open(hash_path, "w") as f:
    f.write(sha256_hash)

print(f"✅ Hash generated: {hash_path.name}")
print(f"🔢 SHA256: {sha256_hash}")