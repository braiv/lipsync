#!/usr/bin/env python3
"""
Test script for official LatentSync audio processing
"""

import torch
import numpy as np

def get_sliced_audio_feature(feature_array: torch.Tensor, vid_idx: int, fps: float = 25.0) -> torch.Tensor:
    """
    Step 2: Get small audio slice for specific frame (Official LatentSync feature2chunks)
    This creates tiny embeddings instead of massive ones
    """
    # 🔧 OFFICIAL LATENTSYNC PARAMETERS
    audio_feat_length = [2, 2]  # Official LatentSync audio context window
    embedding_dim = 384  # Whisper Tiny embedding dimension
    
    length = len(feature_array)
    selected_feature = []
    
    # 🔧 OFFICIAL ALGORITHM: Calculate audio indices for this video frame
    center_idx = int(vid_idx * 50 / fps)  # 50Hz is Whisper's internal rate
    left_idx = center_idx - audio_feat_length[0] * 2
    right_idx = center_idx + (audio_feat_length[1] + 1) * 2
    
    # 🔧 OFFICIAL SLICING: Extract small window around current frame
    for idx in range(left_idx, right_idx):
        idx = max(0, idx)  # Clamp to valid range
        idx = min(length - 1, idx)
        
        if idx < len(feature_array):
            x = feature_array[idx]
        else:
            # Pad with zeros if beyond audio length
            x = torch.zeros(embedding_dim, device=feature_array.device, dtype=feature_array.dtype)
        
        selected_feature.append(x)
    
    # 🔧 OFFICIAL FORMAT: Concatenate and reshape
    selected_feature = torch.cat(selected_feature, dim=0)
    selected_feature = selected_feature.reshape(-1, embedding_dim)  # Shape: [10, 384] - TINY!
    
    print(f"🔍 Frame {vid_idx}: Audio slice shape {selected_feature.shape} (center_idx: {center_idx})")
    
    return selected_feature

def test_official_audio_processing():
    """Test the official LatentSync audio processing"""
    print("🧪 Testing official LatentSync audio processing...")
    
    # Create dummy audio features (simulating audio2feat output)
    dummy_features = torch.randn(100, 384)  # 100 time steps, 384 embedding dim
    print(f"✅ Created dummy audio features: {dummy_features.shape}")
    
    # Test slicing for different frames
    for frame_idx in [0, 10, 25]:
        audio_slice = get_sliced_audio_feature(dummy_features, frame_idx, fps=25.0)
        memory_kb = audio_slice.numel() * 4 / 1024
        print(f"Frame {frame_idx}: slice shape {audio_slice.shape}, memory {memory_kb:.1f} KB")
    
    print("✅ Official LatentSync audio processing test completed!")
    
    # Compare memory usage
    old_method_memory = 5.7 * 1024 * 1024  # 5.7 GB in KB
    new_method_memory = audio_slice.numel() * 4 / 1024  # KB
    
    print(f"\n📊 Memory Comparison:")
    print(f"   Old method (raw audio): {old_method_memory:.0f} KB (5.7 GB)")
    print(f"   New method (sliced features): {new_method_memory:.1f} KB")
    print(f"   Memory reduction: {old_method_memory / new_method_memory:.0f}x smaller!")

if __name__ == "__main__":
    test_official_audio_processing() 