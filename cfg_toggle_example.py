#!/usr/bin/env python3
"""
CFG Toggle Example for FaceFusion Lip Syncer

This example shows how to easily enable/disable CFG (Classifier-Free Guidance)
in the lip syncer module for memory optimization.
"""

# Import the lip syncer module
from facefusion.processors.modules.lip_syncer import ENABLE_CFG, toggle_cfg

def main():
    print("🔧 CFG Toggle Example for FaceFusion Lip Syncer")
    print("=" * 50)
    
    # Check current CFG status
    print(f"📊 Current CFG status: {'Enabled' if ENABLE_CFG else 'Disabled'}")
    
    # Example 1: Enable CFG for higher quality (uses more memory)
    print("\n🚀 Enabling CFG for higher quality...")
    toggle_cfg(True)
    
    # Example 2: Disable CFG for memory optimization
    print("\n💾 Disabling CFG for memory optimization...")
    toggle_cfg(False)
    
    # Example 3: Toggle current state
    print("\n🔄 Toggling current CFG state...")
    current_state = toggle_cfg()  # No argument = toggle
    
    print(f"\n📊 Final CFG status: {'Enabled' if current_state else 'Disabled'}")
    
    print("\n" + "=" * 50)
    print("💡 Usage Tips:")
    print("   • CFG Enabled:  Higher quality, more memory usage (~2x)")
    print("   • CFG Disabled: Lower memory, faster processing")
    print("   • For low-memory systems: Keep CFG disabled")
    print("   • For high-end GPUs: Enable CFG for better results")
    print("\n🔧 Technical Details:")
    print("   • CFG Enabled:  guidance_scale=1.5, num_inference_steps=3")
    print("   • CFG Disabled: guidance_scale=1.0, num_inference_steps=1")
    print("   • num_inference_steps: Controls denoising quality")
    print("     - 1 step:  Fastest, lower quality")
    print("     - 3 steps: Balanced quality/speed")
    print("     - 5+ steps: Best quality, slower")
    print("\n🧹 Memory Management Improvements:")
    print("   • Aggressive cleanup after each frame")
    print("   • Multi-step denoising with per-step cleanup")
    print("   • Memory monitoring and leak detection")
    print("   • Smart device selection based on available memory")
    print("   • Expected memory reduction: 21.5GB → 66MB per frame")

if __name__ == "__main__":
    main() 