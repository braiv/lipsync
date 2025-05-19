import zlib
from pathlib import Path

# ✅ Path to your ONNX model
model_path = Path("latentsync_unet.onnx")

# ✅ Output hash file with just .hash extension
hash_path = model_path.with_suffix("").with_suffix(".hash")

# ✅ Read the file and calculate CRC32
with open(model_path, "rb") as f:
    file_data = f.read()
    crc32_hash = format(zlib.crc32(file_data), '08x')

# ✅ Write the hash to the .hash file
with open(hash_path, "w") as f:
    f.write(crc32_hash)

print(f"✅ Hash generated: {hash_path.name}")
print(f"🔢 CRC32: {crc32_hash}")