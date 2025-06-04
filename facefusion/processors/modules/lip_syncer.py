from argparse import ArgumentParser
from functools import lru_cache
import time
from typing import List

import cv2
import numpy
import torch
import torch.nn.functional as F
import librosa
import scipy.signal
import scipy.io.wavfile
import soundfile as sf
from PIL import Image

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
import facefusion.processors.core as processors
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, inference_manager, logger, process_manager, state_manager, voice_extractor, wording
from facefusion.audio import create_empty_audio_frame, get_voice_frame, read_static_voice, get_raw_audio_frame, create_empty_raw_audio_frame, get_audio_chunks_for_latentsync, get_audio_chunk_from_batch
from facefusion.common_helper import get_first
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.face_helper import create_bounding_box, paste_back, warp_face_by_bounding_box, warp_face_by_face_landmark_5
from facefusion.face_masker import create_mouth_mask, create_occlusion_mask, create_static_box_mask
from facefusion.face_selector import find_similar_faces, sort_and_filter_faces
from facefusion.face_store import get_reference_faces
from facefusion.filesystem import filter_audio_paths, has_audio, in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from facefusion.processors import choices as processors_choices
from facefusion.processors.typing import LipSyncerInputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.typing import ApplyStateItem, Args, AudioFrame, DownloadScope, Face, InferencePool, ModelOptions, ModelSet, ProcessMode, QueuePayload, UpdateProgress, VisionFrame
from facefusion.vision import read_image, read_static_image, restrict_video_fps, write_image

from diffusers.models import AutoencoderKL

# Add LatentSync to Python path
import sys
import os

# Try multiple possible LatentSync locations
latentsync_paths = [
    "/home/cody_braiv_co/latent-sync",
    "../latent-sync",
    "../../latent-sync",
    os.path.expanduser("~/latent-sync"),
    os.path.expanduser("~/braiv-lipsync/latent-sync")
]

for path in latentsync_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.append(path)
        print(f"✅ Added LatentSync path: {path}")
        break

# Import Audio2Feature from the LatentSync package (optional)
try:
    from latentsync.whisper.audio2feature import Audio2Feature
    from latentsync.models.unet import UNet3DConditionModel
    from diffusers import DDIMScheduler
    from omegaconf import OmegaConf
    LATENTSYNC_AVAILABLE = True
except ImportError:
    print("⚠️ LatentSync not available. LatentSync model will be disabled.")
    Audio2Feature = None
    UNet3DConditionModel = None
    DDIMScheduler = None
    OmegaConf = None
    LATENTSYNC_AVAILABLE = False

# Global variables for model caching
audio_encoder = None
vae = None
unet_model = None
scheduler = None

# 🎵 OFFICIAL LATENTSYNC: Global audio features cache
_audio_features_cache = {}  # Cache pre-computed audio features

# 🚀 PERFORMANCE OPTIMIZATION SETTINGS
PERFORMANCE_MODE = "fast_quality"  # 🔧 CRITICAL FIX: Use fast_quality mode for optimal balance of quality and speed
FAST_RESOLUTION = 256  # Use 256x256 for fast mode (matches official benchmarks)
BALANCED_RESOLUTION = 384  # Use 384x384 for balanced mode
QUALITY_RESOLUTION = 512  # Use 512x512 for quality mode

# Performance mode configurations
PERFORMANCE_CONFIGS = {
    "fast": {
        "resolution": 256,
        "num_inference_steps": 10,  # Reduced from 20
        "guidance_scale": 1.0,      # Disable CFG for speed
        "enable_vae_slicing": True,
        "use_fp16": True,
        "batch_optimization": True
    },
    "balanced": {
        "resolution": 384,
        "num_inference_steps": 15,  # Moderate steps
        "guidance_scale": 2.0,      # Light CFG
        "enable_vae_slicing": True,
        "use_fp16": True,
        "batch_optimization": True
    },
    "fast_quality": {
        "resolution": 512,          # Full resolution for quality
        "num_inference_steps": 12,  # Reduced steps for speed
        "guidance_scale": 3.5,      # Full CFG for good lip sync
        "enable_vae_slicing": True, # Enable for memory efficiency
        "use_fp16": True,           # Enable for speed
        "batch_optimization": True  # Enable for efficiency
    },
    "quality": {
        "resolution": 512,
        "num_inference_steps": 20,  # Full steps
        "guidance_scale": 3.5,      # Full CFG
        "enable_vae_slicing": False,
        "use_fp16": False,
        "batch_optimization": False
    }
}

def set_performance_mode(mode: str = "fast_quality"):  # 🔧 CRITICAL FIX: Default to fast_quality for optimal balance
    """
    Set performance mode for LatentSync processing
    
    Args:
        mode: "fast" (2-4 sec/frame), "balanced" (4-8 sec/frame), "fast_quality" (6-12 sec/frame), "quality" (15-30 sec/frame)
    """
    global PERFORMANCE_MODE
    if mode not in PERFORMANCE_CONFIGS:
        print(f"⚠️ Invalid performance mode: {mode}. Using 'fast_quality'")
        mode = "fast_quality"  # 🔧 CRITICAL FIX: Default to fast_quality for optimal balance
    
    PERFORMANCE_MODE = mode
    config = PERFORMANCE_CONFIGS[mode]
    
    print(f"🚀 Performance mode set to: {mode}")
    print(f"   - Resolution: {config['resolution']}x{config['resolution']}")
    print(f"   - Inference steps: {config['num_inference_steps']}")
    print(f"   - Guidance scale: {config['guidance_scale']}")
    print(f"   - FP16: {config['use_fp16']}")
    print(f"   - Expected speed: {get_expected_speed(mode)}")
    
    return config

def get_expected_speed(mode: str) -> str:
    """Get expected processing speed for performance mode"""
    speed_map = {
        "fast": "2-4 sec/frame (matches official benchmarks)",
        "balanced": "4-8 sec/frame (good quality/speed balance)",
        "fast_quality": "6-12 sec/frame (full guidance scale with speed optimizations)",
        "quality": "15-30 sec/frame (maximum quality)"
    }
    return speed_map.get(mode, "unknown")

def get_performance_config():
    """Get current performance configuration"""
    return PERFORMANCE_CONFIGS.get(PERFORMANCE_MODE, PERFORMANCE_CONFIGS["fast_quality"])  # 🔧 CRITICAL FIX: Default to fast_quality

# 🧹 MEMORY MONITORING FUNCTION
def log_memory_usage(stage: str = ""):
    """Log current GPU memory usage for debugging memory leaks."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"💾 {stage} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Warning for high memory usage
        if allocated > 20.0:  # More than 20GB
            print(f"⚠️ HIGH MEMORY USAGE DETECTED: {allocated:.2f}GB")
        elif allocated > 15.0:  # More than 15GB
            print(f"⚠️ Elevated memory usage: {allocated:.2f}GB")
    else:
        print(f"💾 {stage} - CUDA not available")

# CFG toggle flag (global state)
ENABLE_CFG = False  # Default: disabled for memory efficiency

def toggle_cfg(enable: bool = None) -> bool:
    """
    Toggle CFG (Classifier-Free Guidance) on/off.
    
    Args:
        enable: True to enable CFG, False to disable, None to toggle current state
    
    Returns:
        Current CFG state after toggle
    """
    global ENABLE_CFG
    if enable is None:
        ENABLE_CFG = not ENABLE_CFG
    else:
        ENABLE_CFG = enable
    
    status = "enabled" if ENABLE_CFG else "disabled"
    memory_impact = "higher quality, more memory" if ENABLE_CFG else "lower memory, faster"
    print(f"🔧 CFG {status} ({memory_impact})")
    return ENABLE_CFG

# 🧹 MEMORY OPTIMIZATION: Lazy model loading to prevent OOM
# Only load models when needed, not at import time
device = "cuda" if torch.cuda.is_available() else "cpu"

projection_weight = None

def reset_models_to_device(target_device="cuda"):
    """
    Reset all models to a specific device to avoid mixed device states
    Default to GPU for consistency with LatentSync behavior
    """
    global audio_encoder, vae
    
    # Use GPU by default if available, fallback to CPU
    if target_device == "cuda" and not torch.cuda.is_available():
        target_device = "cpu"
        print(f"🔧 CUDA not available, falling back to CPU")
    
    try:
        print(f"🔄 Resetting all models to {target_device}...")
        
        # Reset audio encoder completely
        if audio_encoder is not None and hasattr(audio_encoder, 'model') and audio_encoder.model is not None:
            try:
                print(f"🔧 Resetting audio encoder to {target_device}...")
                audio_encoder.model = audio_encoder.model.to(target_device).float()
                
                # 🔧 CRITICAL: Force ALL parameters to target device
                for name, param in audio_encoder.model.named_parameters():
                    if param.device.type != target_device:
                        param.data = param.data.to(target_device)
                        print(f"🔧 Moved audio encoder parameter {name} to {target_device}")
                
                # 🔧 CRITICAL: Force ALL buffers to target device
                for name, buffer in audio_encoder.model.named_buffers():
                    if buffer.device.type != target_device:
                        buffer.data = buffer.data.to(target_device)
                        print(f"🔧 Moved audio encoder buffer {name} to {target_device}")
                
                print(f"✅ Audio encoder reset to {target_device} with all components")
            except Exception as audio_reset_error:
                print(f"⚠️ Audio encoder reset failed: {audio_reset_error}")
                # Don't set to None - just leave it as is and continue
        
        # Reset VAE completely  
        if vae is not None:
            try:
                if target_device == "cpu":
                    vae = vae.cpu()
                else:
                    vae = vae.to(target_device)
                # Force all parameters to target device
                for param in vae.parameters():
                    param.data = param.data.to(target_device)
                print(f"✅ VAE reset to {target_device}")
            except Exception as vae_reset_error:
                print(f"⚠️ VAE reset failed: {vae_reset_error}")
                # Don't set to None - just leave it as is and continue
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        print(f"✅ All models successfully reset to {target_device}")
        
    except Exception as reset_error:
        print(f"❌ Model reset failed: {reset_error}")
        # 🔧 CRITICAL FIX: Don't delete models on reset failure
        # Just clear CUDA cache and continue - models can still work
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass
        print("⚠️ Continuing with existing models despite reset failure...")


def get_audio_encoder():
    """Lazy loading of Whisper audio encoder"""
    global audio_encoder
    if audio_encoder is None:
        if not LATENTSYNC_AVAILABLE:
            raise RuntimeError("LatentSync is not available. Cannot load audio encoder.")
            
        print("🎵 Loading Whisper Tiny encoder...")
        
        # 🔧 CRITICAL: Use GPU consistently to match LatentSync's behavior
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using {target_device} for audio encoder to ensure device consistency")
            
        try:
            # Use built-in Whisper 'tiny' model and let LatentSync handle device placement
            print(f"🔧 Initializing Audio2Feature with model_path='tiny', device='{target_device}'")
            print(f"🔧 Audio2Feature class: {Audio2Feature}")
            print(f"🔧 LATENTSYNC_AVAILABLE: {LATENTSYNC_AVAILABLE}")
            
            # 🔧 CRITICAL FIX: Use GPU consistently to avoid mixed device conflicts
            audio_encoder = Audio2Feature(model_path="tiny", device=target_device)
            print(f"🔧 Audio2Feature constructor returned: {audio_encoder}")
            print(f"🔧 Audio2Feature type: {type(audio_encoder)}")
            
            # 🔧 CRITICAL: Validate audio encoder was created successfully
            if audio_encoder is None:
                raise RuntimeError("Audio2Feature returned None")
            
            # 🔧 CRITICAL: Ensure audio encoder uses consistent device and float32 precision
            if hasattr(audio_encoder, 'model') and audio_encoder.model is not None:
                print(f"🔧 Audio encoder model device: {audio_encoder.model.device}")
                
                # Ensure float32 precision for stability
                audio_encoder.model = audio_encoder.model.float()
                
                # Ensure ALL parameters are on the same device
                for name, param in audio_encoder.model.named_parameters():
                    if param.device != audio_encoder.model.device:
                        param.data = param.data.to(audio_encoder.model.device)
                        print(f"🔧 Moved parameter {name} to {audio_encoder.model.device}")
                
                # Ensure ALL buffers are on the same device
                for name, buffer in audio_encoder.model.named_buffers():
                    if buffer.device != audio_encoder.model.device:
                        buffer.data = buffer.data.to(audio_encoder.model.device)
                        print(f"🔧 Moved buffer {name} to {audio_encoder.model.device}")
                
                print(f"🔧 Audio encoder set to float32 precision on {audio_encoder.model.device}")
            else:
                print(f"⚠️ Warning: Audio encoder has no model attribute or model is None")
            
            print(f"✅ Audio encoder loaded on {target_device} with device consistency.")
            
            # Test the encoder with a small sample to ensure it works
            print("🧪 Testing audio encoder...")
            test_audio = numpy.zeros(16000, dtype=numpy.float32)  # 1 second of silence at 16kHz
            
            # Create a temporary test file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                test_path = temp_file.name
            
            try:
                sf.write(test_path, test_audio, 16000)
                test_features = audio_encoder.audio2feat(test_path)
                print(f"✅ Audio encoder test successful. Output shape: {test_features.shape}")
            except Exception as test_error:
                print(f"⚠️ Audio encoder test failed: {test_error}")
                # Continue anyway, will handle errors during actual use
            finally:
                if os.path.exists(test_path):
                    os.remove(test_path)
                    
        except Exception as load_error:
            print(f"❌ Failed to load audio encoder: {load_error}")
            print(f"❌ Load error type: {type(load_error)}")
            import traceback
            print(f"❌ Load error traceback: {traceback.format_exc()}")
            # 🔧 CRITICAL FIX: Set audio_encoder to None on failure so it can be retried
            audio_encoder = None
            raise RuntimeError(f"Could not initialize Audio2Feature: {load_error}")
    else:
        # 🔧 CRITICAL: Validate existing audio encoder
        if audio_encoder is None:
            print("⚠️ Audio encoder is None, attempting to reload...")
            # Recursive call to reload
            audio_encoder = None  # Reset to trigger reload
            return get_audio_encoder()
        
        # 🔧 CRITICAL: Ensure audio encoder stays on consistent device
        if hasattr(audio_encoder, 'model') and audio_encoder.model is not None:
            try:
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
                current_device = audio_encoder.model.device
                
                # Only move if device is different
                if current_device.type != target_device:
                    print(f"🔧 Moving audio encoder from {current_device} to {target_device}")
                    audio_encoder.model = audio_encoder.model.to(target_device).float()
                    
                    # Ensure ALL parameters are on target device
                    for name, param in audio_encoder.model.named_parameters():
                        if param.device.type != target_device:
                            param.data = param.data.to(target_device)
                            print(f"🔧 Moved audio encoder parameter {name} to {target_device}")
                    
                    # Ensure ALL buffers are on target device
                    for name, buffer in audio_encoder.model.named_buffers():
                        if buffer.device.type != target_device:
                            buffer.data = buffer.data.to(target_device)
                            print(f"🔧 Moved audio encoder buffer {name} to {target_device}")
                
                print(f"🔧 Audio encoder confirmed on {audio_encoder.model.device}")
            except Exception as move_error:
                print(f"⚠️ Audio encoder device consistency check failed: {move_error}")
    
    # 🔧 FINAL VALIDATION: Ensure we return a valid encoder
    if audio_encoder is None:
        raise RuntimeError("Audio encoder is still None after initialization attempt")
    
    return audio_encoder

def get_vae():
    """Lazy loading of VAE model"""
    global vae
    if vae is None:
        print("🖼️ Loading VAE model...")
        
        # 🔧 CRITICAL: Use same device logic as audio encoder for consistency
        if torch.cuda.is_available():
            vae_device = "cuda"
            print(f"✅ Using GPU for VAE (consistent with audio encoder)")
        else:
            vae_device = "cpu"
            print(f"⚠️ Using CPU for VAE (CUDA not available)")
            
        # 🔧 CRITICAL FIX: Use correct VAE model and configuration (matching official LatentSync)
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(vae_device)
        
        # 🔧 CRITICAL FIX: Set VAE to FP16 for consistency with performance mode
        perf_config = get_performance_config()
        if perf_config["use_fp16"] and torch.cuda.is_available():
            vae = vae.half()  # Convert VAE to FP16
            print(f"🚀 VAE converted to FP16 for performance consistency")
        else:
            vae = vae.float()  # Keep as FP32
            print(f"🔧 VAE kept in FP32 precision")
        
        vae = vae.eval()
        
        # 🔧 CRITICAL: Set official LatentSync VAE configuration
        vae.config.scaling_factor = 0.18215  # Official LatentSync scaling factor
        vae.config.shift_factor = 0          # Official LatentSync shift factor
        
        print(f"✅ VAE loaded with official LatentSync config:")
        print(f"   - Model: stabilityai/sd-vae-ft-mse (official)")
        print(f"   - Scaling factor: {vae.config.scaling_factor}")
        print(f"   - Shift factor: {vae.config.shift_factor}")
        print(f"   - Device: {vae_device}")
        print(f"   - Precision: {'FP16' if vae.dtype == torch.float16 else 'FP32'}")
    else:
        # 🔧 CRITICAL: Ensure VAE precision matches performance mode
        perf_config = get_performance_config()
        target_dtype = torch.float16 if perf_config["use_fp16"] and torch.cuda.is_available() else torch.float32
        
        if vae.dtype != target_dtype:
            print(f"🔧 Converting VAE from {vae.dtype} to {target_dtype} for consistency")
            if target_dtype == torch.float16:
                vae = vae.half()
            else:
                vae = vae.float()
    
    return vae

def verify_device_consistency():
    """
    Verify that audio encoder and VAE are on the same device.
    Returns the common device if consistent, raises error if not.
    """
    global audio_encoder, vae
    
    devices = []
    
    # Check audio encoder device
    if audio_encoder is not None and hasattr(audio_encoder, 'model') and audio_encoder.model is not None:
        audio_device = audio_encoder.model.device
        devices.append(('audio_encoder', audio_device))
        print(f"🔍 Audio encoder device: {audio_device}")
    else:
        print("⚠️ Audio encoder not available for device check")
    
    # Check VAE device
    if vae is not None:
        vae_device = vae.device
        devices.append(('vae', vae_device))
        print(f"🔍 VAE device: {vae_device}")
    else:
        print("⚠️ VAE not available for device check")
    
    if len(devices) < 2:
        print("⚠️ Cannot verify device consistency - not all models loaded")
        return None
    
    # Check if all devices are the same
    device_types = set(device.type for _, device in devices)
    
    if len(device_types) == 1:
        common_device = devices[0][1].type
        print(f"✅ Device consistency verified: all models on {common_device}")
        return common_device
    else:
        device_info = ", ".join(f"{name}: {device}" for name, device in devices)
        raise RuntimeError(f"❌ Device inconsistency detected! {device_info}")

def get_projection_weight():
    """Lazy loading of projection weight"""
    global projection_weight
    if projection_weight is None:
        projection_weight = torch.randn(384, 4).float().to("cuda" if torch.cuda.is_available() else "cpu")  # FP32
    return projection_weight

def clear_models():
    """Clear all models from memory"""
    global audio_encoder, vae, projection_weight
    if audio_encoder is not None:
        del audio_encoder
        audio_encoder = None
    if vae is not None:
        del vae
        vae = None
    if projection_weight is not None:
        del projection_weight
        projection_weight = None
    torch.cuda.empty_cache()
    print("🧹 All LatentSync models cleared from memory.")

@lru_cache(maxsize = None)
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'wav2lip_96':
		{
			'hashes':
			{
				'lip_syncer':
				{
					'url': resolve_download_url('models-3.0.0', 'wav2lip_96.hash'),
					'path': resolve_relative_path('../.assets/models/wav2lip_96.hash')
				}
			},
			'sources':
			{
				'lip_syncer':
				{
					'url': resolve_download_url('models-3.0.0', 'wav2lip_96.onnx'),
					'path': resolve_relative_path('../.assets/models/wav2lip_96.onnx')
				}
			},
			'size': (96, 96)
		},
		'wav2lip_gan_96':
		{
			'hashes':
			{
				'lip_syncer':
				{
					'url': resolve_download_url('models-3.0.0', 'wav2lip_gan_96.hash'),
					'path': resolve_relative_path('../.assets/models/wav2lip_gan_96.hash')
				}
			},
			'sources':
			{
				'lip_syncer':
				{
					'url': resolve_download_url('models-3.0.0', 'wav2lip_gan_96.onnx'),
					'path': resolve_relative_path('../.assets/models/wav2lip_gan_96.onnx')
				}
			},
			'size': (96, 96)
		},
		'latentsync':
		{
			'hashes':
			{
				'lip_syncer':
				{
					'url': None,  # No hash needed for PyTorch models
					'path': None
				}
			},
			'sources':
			{
				'lip_syncer':
				{
					'url': None,  # No download needed - using PyTorch models directly
					'path': None  # PyTorch models loaded from LatentSync installation
				}
			},
			'size': (512, 512)
		}
	}


def get_inference_pool() -> InferencePool:
	model_sources = get_model_options().get('sources')
	# LatentSync uses PyTorch models directly, not ONNX inference
	if state_manager.get_item('lip_syncer_model') == 'latentsync':
		return {}  # Return empty pool for LatentSync
	return inference_manager.get_inference_pool(__name__, model_sources)


def clear_inference_pool() -> None:
	inference_manager.clear_inference_pool(__name__)


def get_model_options() -> ModelOptions:
	lip_syncer_model = state_manager.get_item('lip_syncer_model')
	return create_static_model_set('full').get(lip_syncer_model)


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--lip-syncer-model', help = wording.get('help.lip_syncer_model'), default = config.get_str_value('processors.lip_syncer_model', 'wav2lip_gan_96'), choices = processors_choices.lip_syncer_models)
		facefusion.jobs.job_store.register_step_keys([ 'lip_syncer_model' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('lip_syncer_model', args.get('lip_syncer_model'))


def pre_check() -> bool:
	# LatentSync uses PyTorch models directly - no download validation needed
	if state_manager.get_item('lip_syncer_model') == 'latentsync':
		return True
	
	# For other models (Wav2Lip), perform normal download validation
	model_hashes = get_model_options().get('hashes')
	model_sources = get_model_options().get('sources')

	return conditional_download_hashes(model_hashes) and conditional_download_sources(model_sources)


def pre_process(mode : ProcessMode) -> bool:
	if not has_audio(state_manager.get_item('source_paths')):
		logger.error(wording.get('choose_audio_source') + wording.get('exclamation_mark'), __name__)
		return False
	if mode in [ 'output', 'preview' ] and not is_image(state_manager.get_item('target_path')) and not is_video(state_manager.get_item('target_path')):
		logger.error(wording.get('choose_image_or_video_target') + wording.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
		logger.error(wording.get('specify_image_or_video_output') + wording.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not same_file_extension([ state_manager.get_item('target_path'), state_manager.get_item('output_path') ]):
		logger.error(wording.get('match_target_and_output_extension') + wording.get('exclamation_mark'), __name__)
		return False
	return True


def post_process() -> None:
	read_static_image.cache_clear()
	read_static_voice.cache_clear()
	if state_manager.get_item('video_memory_strategy') in [ 'strict', 'moderate' ]:
		clear_inference_pool()
	if state_manager.get_item('video_memory_strategy') == 'strict':
		content_analyser.clear_inference_pool()
		face_classifier.clear_inference_pool()
		face_detector.clear_inference_pool()
		face_landmarker.clear_inference_pool()
		face_masker.clear_inference_pool()
		face_recognizer.clear_inference_pool()
		voice_extractor.clear_inference_pool()
		# 🧹 Clear LatentSync models from memory
		clear_models()
		# 🧹 Clear audio features cache
		clear_audio_features_cache()


def forward(temp_audio_frame: AudioFrame, close_vision_frame: VisionFrame, target_face: Face = None) -> VisionFrame:
    """
    Simplified forward function that uses the official LatentSync pipeline
    """
    model_name = state_manager.get_item('lip_syncer_model')
    
    if model_name == 'latentsync':
        # 🔧 CRITICAL FIX: AudioFrame contains mel-spectrogram, not raw audio!
        # LatentSync needs raw audio waveform, not mel-spectrogram
        
        print(f"🔧 DEBUG - AudioFrame analysis:")
        print(f"   - Type: {type(temp_audio_frame)}")
        print(f"   - Shape: {temp_audio_frame.shape if hasattr(temp_audio_frame, 'shape') else 'No shape'}")
        
        # 🚨 CRITICAL: AudioFrame is mel-spectrogram for Wav2Lip, but LatentSync needs raw audio
        # We need to get the original raw audio instead of using the processed mel-spectrogram
        
        if isinstance(temp_audio_frame, torch.Tensor) and temp_audio_frame.dim() == 2 and temp_audio_frame.shape[1] == 384:
            # This is a pre-computed audio slice from the official method - pass it directly
            print(f"✅ Forward: Using pre-computed audio slice {temp_audio_frame.shape}")
            audio_input = temp_audio_frame
        elif isinstance(temp_audio_frame, numpy.ndarray) and temp_audio_frame.ndim == 1:
            # This is raw audio waveform - perfect for LatentSync
            print(f"✅ Forward: Using raw audio waveform {temp_audio_frame.shape}")
            audio_input = temp_audio_frame
        else:
            # 🚨 CRITICAL ISSUE: AudioFrame is mel-spectrogram, not raw audio!
            print("🚨 CRITICAL: AudioFrame contains mel-spectrogram, not raw audio!")
            print("🔧 LatentSync requires raw audio waveform, not mel-spectrogram")
            print("🔧 This explains why audio input is silent - we're processing the wrong data!")
            
            # 🔧 CRITICAL FIX: Try to get raw audio from source
            try:
                # Get the source audio path from state manager
                source_paths = state_manager.get_item('source_paths', [])
                source_audio_path = None
                
                for path in source_paths:
                    if has_audio(path):
                        source_audio_path = path
                        break
                
                if source_audio_path:
                    print(f"🔧 Extracting raw audio from: {source_audio_path}")
                    
                    # 🔧 DEBUG: Check if file exists and get basic info
                    if not os.path.exists(source_audio_path):
                        print(f"❌ Audio file does not exist: {source_audio_path}")
                        raise FileNotFoundError(f"Audio file not found: {source_audio_path}")
                    
                    file_size = os.path.getsize(source_audio_path)
                    print(f"🔧 Audio file info: size={file_size} bytes, exists={os.path.exists(source_audio_path)}")
                    
                    # Read raw audio at 16kHz for LatentSync
                    try:
                        audio_data, sample_rate = sf.read(source_audio_path)
                        print(f"🔧 Raw audio loaded: shape={audio_data.shape}, sample_rate={sample_rate}, dtype={audio_data.dtype}")
                        print(f"🔧 Raw audio stats: mean={audio_data.mean():.6f}, std={audio_data.std():.6f}, range=[{audio_data.min():.6f}, {audio_data.max():.6f}]")
                        
                        # Check if audio is actually silent
                        audio_rms = numpy.sqrt(numpy.mean(audio_data**2))
                        print(f"🔧 Raw audio RMS: {audio_rms:.6f}")
                        
                        if audio_rms < 0.001:
                            print(f"⚠️ WARNING: Source audio file appears to be silent!")
                            print(f"🔧 This may be the root cause of the lip sync issue")
                        
                    except Exception as read_error:
                        print(f"❌ Failed to read audio file: {read_error}")
                        raise read_error
                    
                    # Ensure mono
                    if len(audio_data.shape) > 1:
                        print(f"🔧 Converting from {audio_data.shape} to mono")
                        audio_data = numpy.mean(audio_data, axis=1)
                    
                    # Resample to 16kHz if needed
                    if sample_rate != 16000:
                        print(f"🔧 Resampling from {sample_rate}Hz to 16000Hz")
                        audio_data = resample_audio(audio_data, sample_rate, 16000)
                    
                    # Use a reasonable chunk size (1 second)
                    chunk_size = 16000
                    if len(audio_data) > chunk_size:
                        # Take first second of audio
                        print(f"🔧 Truncating audio from {len(audio_data)} to {chunk_size} samples")
                        audio_data = audio_data[:chunk_size]
                    elif len(audio_data) < chunk_size:
                        # Pad with zeros if too short
                        print(f"🔧 Padding audio from {len(audio_data)} to {chunk_size} samples")
                        audio_data = numpy.pad(audio_data, (0, chunk_size - len(audio_data)), mode='constant')
                    
                    audio_input = audio_data.astype(numpy.float32)
                    
                    print(f"✅ Successfully extracted raw audio:")
                    print(f"   - Shape: {audio_input.shape}")
                    print(f"   - Duration: {len(audio_input) / 16000:.2f} seconds")
                    print(f"   - RMS: {numpy.sqrt(numpy.mean(audio_input**2)):.6f}")
                    
                    # Verify audio is not silent
                    if numpy.sqrt(numpy.mean(audio_input**2)) < 0.001:
                        print(f"⚠️ WARNING: Extracted audio appears silent!")
                        print(f"🔧 This confirms the source audio file is silent or very quiet")
                        print(f"🔧 Lip sync will not work with silent audio")
                    else:
                        print(f"✅ Audio extraction successful - non-silent audio detected")
                        
                else:
                    raise RuntimeError("No audio source found in source_paths")
                    
            except Exception as audio_error:
                print(f"❌ Raw audio extraction failed: {audio_error}")
                print(f"🔧 Creating dummy audio - NO LIP SYNC will occur!")
                audio_input = numpy.zeros(16000, dtype=numpy.float32)  # 1 second of silence
        
        # Pass the audio input to process_frame_latentsync for proper handling
        return process_frame_latentsync(target_face, close_vision_frame, audio_input)
    
    else:
        # For other models (like Wav2Lip), use ONNX inference
        lip_syncer = get_inference_pool().get('lip_syncer')
        
        with conditional_thread_semaphore():
            close_vision_frame = lip_syncer.run(None, {
                'source': temp_audio_frame,
                'target': close_vision_frame
            })[0]
            
            close_vision_frame = normalize_close_frame(close_vision_frame)
        
        return close_vision_frame


def prepare_audio_frame(temp_audio_frame : AudioFrame) -> AudioFrame:
	temp_audio_frame = numpy.maximum(numpy.exp(-5 * numpy.log(10)), temp_audio_frame)
	temp_audio_frame = numpy.log10(temp_audio_frame) * 1.6 + 3.2
	temp_audio_frame = temp_audio_frame.clip(-4, 4).astype(numpy.float32)
	temp_audio_frame = numpy.expand_dims(temp_audio_frame, axis = (0, 1))
	return temp_audio_frame


def prepare_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis = 0)
	prepare_vision_frame = crop_vision_frame.copy()
	prepare_vision_frame[:, 48:] = 0
	crop_vision_frame = numpy.concatenate((prepare_vision_frame, crop_vision_frame), axis = 3)
	crop_vision_frame = crop_vision_frame.transpose(0, 3, 1, 2).astype('float32') / 255.0
	return crop_vision_frame


def normalize_close_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
    crop_vision_frame = crop_vision_frame[0].transpose(1, 2, 0)
    crop_vision_frame = crop_vision_frame.clip(0, 1) * 255
    crop_vision_frame = crop_vision_frame.astype(numpy.uint8)
    return crop_vision_frame


def resample_audio(audio_waveform: numpy.ndarray, original_sample_rate: int, target_sample_rate: int) -> numpy.ndarray:
    """
    Resample audio waveform from original sample rate to target sample rate.
    
    :param audio_waveform: Input audio waveform as numpy array
    :param original_sample_rate: Original sample rate of the audio
    :param target_sample_rate: Target sample rate (e.g., 16000 for Whisper)
    :return: Resampled audio waveform
    """
    try:
        if original_sample_rate == target_sample_rate:
            return audio_waveform
        
        # Use librosa for high-quality resampling
        resampled_audio = librosa.resample(
            y=audio_waveform,
            orig_sr=original_sample_rate,
            target_sr=target_sample_rate,
            res_type='kaiser_best'  # High-quality resampling
        )
        
        return resampled_audio.astype(numpy.float32)
        
    except Exception as e:
        # Fallback to scipy if librosa fails
        try:
            # Calculate the resampling ratio
            ratio = target_sample_rate / original_sample_rate
            num_samples = int(len(audio_waveform) * ratio)
            
            # Use scipy's resample function
            resampled_audio = scipy.signal.resample(audio_waveform, num_samples)
            return resampled_audio.astype(numpy.float32)
            
        except Exception as fallback_error:
            print(f"⚠️ Resampling failed with both librosa and scipy: {e}, {fallback_error}")
            print(f"⚠️ Returning original audio without resampling")
            return audio_waveform.astype(numpy.float32)


def get_reference_frame(source_face : Face, target_face : Face, temp_vision_frame : VisionFrame) -> VisionFrame:
	pass


def process_frame_latentsync(source_face: Face, target_frame: VisionFrame, audio_chunk) -> VisionFrame:
    """
    Process a single frame using the official LatentSync pipeline with performance optimizations
    """
    global audio_encoder, vae, unet_model, scheduler
    
    # 🚀 Get performance configuration
    perf_config = get_performance_config()
    target_resolution = perf_config["resolution"]
    num_inference_steps = perf_config["num_inference_steps"]
    guidance_scale = perf_config["guidance_scale"]
    use_fp16 = perf_config["use_fp16"]
    
    # 🧹 Memory monitoring
    log_memory_usage("🎬 Starting frame processing")
    
    # Get model name from global state
    model_name = state_manager.get_item('lip_syncer_model')
    target_device = get_device_for_lip_syncer()
    
    try:
        if model_name == 'latentsync':
            # ===== OPTIMIZED LATENTSYNC PIPELINE =====
            
            # 1. Load models with performance optimizations
            audio_encoder = get_audio_encoder()
            vae = get_vae()
            unet_model = get_unet_model()
            scheduler = get_scheduler()
            
            # 🚀 PERFORMANCE: Enable VAE slicing for memory efficiency
            if perf_config["enable_vae_slicing"] and hasattr(vae, 'enable_slicing'):
                vae.enable_slicing()
            
            # 🚀 PERFORMANCE: Set optimal dtype based on performance mode
            if use_fp16 and torch.cuda.is_available():
                target_dtype = torch.float16
                print(f"🚀 Using FP16 for {target_resolution}x{target_resolution} processing")
            else:
                target_dtype = torch.float32
                print(f"🔧 Using FP32 for {target_resolution}x{target_resolution} processing")
            
            # 2. Pipeline parameters (optimized based on performance mode)
            do_classifier_free_guidance = guidance_scale > 1.0
            
            print(f"🚀 Performance mode: {PERFORMANCE_MODE}")
            print(f"   - Resolution: {target_resolution}x{target_resolution}")
            print(f"   - Steps: {num_inference_steps}")
            print(f"   - CFG: {guidance_scale}")
            print(f"   - Precision: {'FP16' if use_fp16 else 'FP32'}")
            
            # 3. Set scheduler timesteps
            scheduler.set_timesteps(num_inference_steps, device=target_device)
            timesteps = scheduler.timesteps
            
            # 4. Process audio and image in proper scope
            with torch.no_grad():
                # 🔧 CRITICAL FIX: Handle both pre-computed slices and raw audio
                print("🎵 Processing audio...")
                
                # 🔧 CRITICAL DEBUG: Check audio input quality
                print(f"🔧 DEBUG - Audio input analysis:")
                if isinstance(audio_chunk, torch.Tensor):
                    print(f"   - Type: torch.Tensor")
                    print(f"   - Shape: {audio_chunk.shape}")
                    print(f"   - Dtype: {audio_chunk.dtype}")
                    if audio_chunk.dim() == 2 and audio_chunk.shape[1] == 384:
                        print(f"   - Pre-computed audio slice detected")
                        print(f"   - Mean: {audio_chunk.mean():.4f}, Std: {audio_chunk.std():.4f}")
                        print(f"   - Range: [{audio_chunk.min():.4f}, {audio_chunk.max():.4f}]")
                    else:
                        print(f"   - Raw audio tensor")
                        if audio_chunk.numel() > 0:
                            print(f"   - Mean: {audio_chunk.mean():.4f}, Std: {audio_chunk.std():.4f}")
                
                if isinstance(audio_chunk, torch.Tensor) and audio_chunk.dim() == 2 and audio_chunk.shape[1] == 384:
                    # This is a pre-computed audio slice - use directly
                    print(f"✅ Using pre-computed audio slice: {audio_chunk.shape}")
                    audio_embeds = encode_audio_for_latentsync(audio_chunk)
                else:
                    # This is raw audio - process through fallback path
                    print("⚠️ Processing raw audio through fallback path")
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_chunk = audio_chunk.cpu().numpy()
                    
                    if len(audio_chunk.shape) == 1:
                        audio_chunk = audio_chunk[None, :]
                    
                    # 🔧 CRITICAL DEBUG: Check raw audio quality
                    print(f"🔧 DEBUG - Raw audio analysis:")
                    print(f"   - Shape: {audio_chunk.shape}")
                    print(f"   - Duration: {audio_chunk.shape[-1] / 16000:.2f} seconds")
                    print(f"   - Mean: {audio_chunk.mean():.4f}, Std: {audio_chunk.std():.4f}")
                    print(f"   - Range: [{audio_chunk.min():.4f}, {audio_chunk.max():.4f}]")
                    
                    # Check if audio is silent or very quiet
                    audio_rms = numpy.sqrt(numpy.mean(audio_chunk**2))
                    print(f"   - RMS: {audio_rms:.6f}")
                    
                    if audio_rms < 0.001:
                        print("⚠️ WARNING: Audio appears to be silent or very quiet!")
                        print("🔧 This may result in no lip movement")
                    elif audio_rms > 0.1:
                        print("⚠️ WARNING: Audio appears to be very loud!")
                        print("🔧 This may cause excessive lip movement")
                    else:
                        print("✅ Audio level appears normal")
                    
                    # 🔧 CRITICAL FIX: Ensure audio chunk is reasonable size for single frame
                    max_audio_samples = 16000 * 2  # 2 seconds max for single frame
                    if audio_chunk.shape[-1] > max_audio_samples:
                        print(f"⚠️ Audio chunk too large: {audio_chunk.shape} samples")
                        print(f"🔧 Truncating to {max_audio_samples} samples for memory efficiency")
                        audio_chunk = audio_chunk[:, :max_audio_samples]
                    
                    # 🔧 MEMORY SAFETY: Pre-check audio size to prevent OOM in audio encoder
                    audio_duration = audio_chunk.shape[-1] / 16000  # Duration in seconds
                    estimated_memory_gb = audio_duration * 2.0  # Rough estimate: 2GB per second of audio
                    
                    if estimated_memory_gb > 4.0:  # If estimated > 4GB, use dummy embeddings
                        print(f"⚠️ Audio too large for encoding (estimated {estimated_memory_gb:.1f}GB)")
                        print("🔧 Using dummy embeddings to prevent OOM")
                        print("⚠️ WARNING: Dummy embeddings will result in NO lip sync!")
                        
                        # Create safe dummy embeddings
                        batch_size = 1
                        seq_len = 77  # Standard sequence length
                        embed_dim = 384  # Whisper Tiny embedding dimension
                        audio_embeds = torch.zeros(batch_size, seq_len, embed_dim, device=target_device)
                    
                    else:
                        # Convert to tensor and get audio features
                        audio_tensor = torch.from_numpy(audio_chunk).float().to(target_device)
                        
                        # 🔧 SAFE AUDIO ENCODING: Try encoding with OOM protection
                        try:
                            # Get audio embeddings (matching official whisper processing)
                            audio_embeds = encode_audio_for_latentsync(audio_tensor)
                            
                            # Verify embeddings are valid
                            if audio_embeds is None:
                                raise RuntimeError("Audio encoding returned None")
                                
                        except (RuntimeError, torch.cuda.OutOfMemoryError) as encoding_error:
                            print(f"⚠️ Audio encoding failed: {encoding_error}")
                            print("🔧 Falling back to dummy embeddings")
                            
                            # Clear CUDA cache and create dummy embeddings
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # Create safe dummy embeddings
                            batch_size = 1
                            seq_len = 77  # Standard sequence length
                            embed_dim = 384  # Whisper Tiny embedding dimension
                            audio_embeds = torch.zeros(batch_size, seq_len, embed_dim, device=target_device)
                
                # 🔧 MEMORY CHECK: Validate audio embeddings size
                if audio_embeds is not None:
                    audio_memory_gb = audio_embeds.numel() * audio_embeds.element_size() / (1024**3)
                    print(f"🔧 Audio embeddings memory: {audio_memory_gb:.2f} GB")
                    
                    # 🔧 CRITICAL DEBUG: Analyze audio embeddings quality
                    print(f"🔧 DEBUG - Audio embeddings analysis:")
                    print(f"   - Shape: {audio_embeds.shape}")
                    print(f"   - Dtype: {audio_embeds.dtype}")
                    print(f"   - Mean: {audio_embeds.mean():.6f}, Std: {audio_embeds.std():.6f}")
                    print(f"   - Range: [{audio_embeds.min():.6f}, {audio_embeds.max():.6f}]")
                    
                    # Check if embeddings are meaningful (not all zeros or constant)
                    if torch.all(audio_embeds == 0):
                        print("⚠️ CRITICAL: Audio embeddings are all zeros!")
                        print("🔧 This will result in NO lip sync movement")
                    elif audio_embeds.std() < 0.001:
                        print("⚠️ WARNING: Audio embeddings have very low variance!")
                        print("🔧 This may result in minimal lip sync movement")
                    else:
                        print("✅ Audio embeddings appear to have meaningful content")
                    
                    # 🔧 CRITICAL FIX: Reshape audio embeddings for UNet compatibility
                    # The UNet expects [batch, seq_len, embed_dim] format
                    if audio_embeds.dim() == 3 and audio_embeds.shape[0] > 1:
                        # If we have [seq_len, frames, embed_dim], reshape to [batch, seq_len*frames, embed_dim]
                        batch_size = 1
                        seq_len = audio_embeds.shape[0] * audio_embeds.shape[1]  # 50 * 5 = 250
                        embed_dim = audio_embeds.shape[2]  # 384
                        audio_embeds = audio_embeds.reshape(batch_size, seq_len, embed_dim)
                        print(f"🔧 Reshaped audio embeddings from [50, 5, 384] to {audio_embeds.shape}")
                    
                    # If embeddings are too large, truncate them
                    if audio_memory_gb > 2.0:  # More than 2GB is too much for single frame
                        print(f"⚠️ Audio embeddings too large ({audio_memory_gb:.2f} GB), truncating...")
                        max_seq_len = min(audio_embeds.shape[1], 1500)  # Reasonable sequence length
                        audio_embeds = audio_embeds[:, :max_seq_len, :]
                        print(f"🔧 Truncated to: {audio_embeds.shape}")
                
                if audio_embeds.dim() == 2:
                    audio_embeds = audio_embeds.unsqueeze(0)  # Add batch dim
                
                # 🚀 PERFORMANCE: Convert audio embeddings to target dtype
                audio_embeds = audio_embeds.to(dtype=target_dtype)
                print(f"🔧 Audio embeddings dtype: {audio_embeds.dtype}")
                print(f"🔧 Final audio embeddings shape: {audio_embeds.shape}")
                
                # 🚀 PERFORMANCE: Optimized CFG handling
                if do_classifier_free_guidance:
                    audio_memory_gb = audio_embeds.numel() * audio_embeds.element_size() / (1024**3)
                    if audio_memory_gb > 1.0:  # If > 1GB, disable CFG to save memory
                        print(f"⚠️ Disabling CFG due to large audio embeddings ({audio_memory_gb:.2f} GB)")
                        do_classifier_free_guidance = False
                        guidance_scale = 1.0
                    else:
                        # Create unconditional (empty) audio embedding
                        uncond_audio_embeds = torch.zeros_like(audio_embeds)
                        audio_embeds = torch.cat([uncond_audio_embeds, audio_embeds])
                        print(f"🔧 CFG enabled - doubled embeddings to: {audio_embeds.shape}")
                        
                        # 🔧 CRITICAL DEBUG: Check CFG embeddings
                        print(f"🔧 DEBUG - CFG embeddings analysis:")
                        print(f"   - Unconditional part mean: {audio_embeds[:1].mean():.6f}")
                        print(f"   - Conditional part mean: {audio_embeds[1:].mean():.6f}")
                        print(f"   - Difference: {(audio_embeds[1:] - audio_embeds[:1]).mean():.6f}")
                
                log_memory_usage("🎵 Audio processing complete")
                
                # 5. Process target frame with optimized resolution
                print(f"🖼️ Processing target frame at {target_resolution}x{target_resolution}...")
                target_frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
                target_pil = Image.fromarray(target_frame_rgb)
                
                # 🚀 PERFORMANCE: Use optimized resolution based on performance mode
                target_pil = target_pil.resize((target_resolution, target_resolution), Image.LANCZOS)
                
                # Convert to tensor and normalize
                target_tensor = torch.from_numpy(numpy.array(target_pil)).float() / 255.0
                target_tensor = target_tensor.permute(2, 0, 1).unsqueeze(0).to(target_device)
                target_tensor = (target_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
                
                # 🚀 PERFORMANCE: Convert target tensor to target dtype
                target_tensor = target_tensor.to(dtype=target_dtype)
                
                # 6. Encode image to latents
                print("🔄 Encoding image to latents...")
                image_latents = vae.encode(target_tensor).latent_dist.sample()
                # 🔧 CRITICAL FIX: Use official LatentSync VAE scaling (with shift factor)
                image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
                
                # 🚀 PERFORMANCE: Ensure latents are in target dtype
                image_latents = image_latents.to(dtype=target_dtype)
                
                # 7. Create mask (mouth region) with optimized resolution
                print("🎭 Creating mouth mask...")
                mask = create_latentsync_mouth_mask(target_frame, source_face)
                mask_pil = Image.fromarray((mask * 255).astype(numpy.uint8)).resize((target_resolution, target_resolution), Image.LANCZOS)
                mask_tensor = torch.from_numpy(numpy.array(mask_pil)).float() / 255.0
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(target_device)  # [1, 1, res, res]
                
                # 🚀 PERFORMANCE: Convert mask to target dtype
                mask_tensor = mask_tensor.to(dtype=target_dtype)
                
                # Encode mask to latent space (adjust size based on resolution)
                latent_size = target_resolution // 8  # VAE downsamples by 8x
                mask_latents = F.interpolate(mask_tensor, size=(latent_size, latent_size), mode='nearest')
                mask_latents = mask_latents.unsqueeze(2)  # Add frame dimension [1, 1, 1, latent_size, latent_size]
                
                # 8. Create masked image latents
                print("🖼️ Creating masked image latents...")
                masked_image = target_tensor * (1 - mask_tensor)
                masked_image_latents = vae.encode(masked_image).latent_dist.sample()
                # 🔧 CRITICAL FIX: Use official LatentSync VAE scaling (with shift factor)
                masked_image_latents = (masked_image_latents - vae.config.shift_factor) * vae.config.scaling_factor
                masked_image_latents = masked_image_latents.unsqueeze(2)  # Add frame dimension
                
                # 🚀 PERFORMANCE: Ensure masked image latents are in target dtype
                masked_image_latents = masked_image_latents.to(dtype=target_dtype)
                
                # 9. Prepare reference latents (same as image latents for single frame)
                ref_latents = image_latents.unsqueeze(2)  # Add frame dimension
                
                # 10. Prepare initial noise latents with optimized size
                print("🎲 Preparing initial latents...")
                batch_size = 1
                num_frames = 1
                num_channels_latents = 4
                height, width = latent_size, latent_size  # Use calculated latent size
                
                latents = prepare_latents(
                    batch_size, num_frames, num_channels_latents, height, width,
                    dtype=target_dtype, device=target_device  # 🚀 Use target dtype
                )
                
                log_memory_usage("🔄 Latents prepared")
                
                # 11. OPTIMIZED DENOISING LOOP
                print(f"🚀 Starting {num_inference_steps}-step denoising (optimized)...")
                
                for i, t in enumerate(timesteps):
                    print(f"  Step {i+1}/{num_inference_steps} (t={t})")
                    
                    # Expand latents for CFG
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    
                    # Scale model input (official scheduler scaling)
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                    
                    # 🚀 PERFORMANCE: Ensure latent input is in target dtype
                    latent_model_input = latent_model_input.to(dtype=target_dtype)
                    
                    # Prepare conditioning inputs
                    if do_classifier_free_guidance:
                        # Duplicate conditioning for CFG
                        mask_input = torch.cat([mask_latents] * 2)
                        masked_img_input = torch.cat([masked_image_latents] * 2)
                        ref_input = torch.cat([ref_latents] * 2)
                    else:
                        mask_input = mask_latents
                        masked_img_input = masked_image_latents
                        ref_input = ref_latents
                    
                    # 🚀 PERFORMANCE: Ensure all conditioning inputs are in target dtype
                    mask_input = mask_input.to(dtype=target_dtype)
                    masked_img_input = masked_img_input.to(dtype=target_dtype)
                    ref_input = ref_input.to(dtype=target_dtype)
                    
                    # Concatenate all inputs (official order)
                    print(f"🔧 UNet input shapes before concatenation:")
                    print(f"   - latent_model_input: {latent_model_input.shape}")
                    print(f"   - mask_input: {mask_input.shape}")
                    print(f"   - masked_img_input: {masked_img_input.shape}")
                    print(f"   - ref_input: {ref_input.shape}")
                    
                    unet_input = torch.cat([
                        latent_model_input,
                        mask_input,
                        masked_img_input,
                        ref_input
                    ], dim=1)
                    
                    print(f"🔧 Final UNet input shape: {unet_input.shape}")
                    print(f"🔧 Audio embeddings shape: {audio_embeds.shape}")
                    
                    # 🚀 PERFORMANCE: Final dtype check before UNet forward
                    unet_input = unet_input.to(dtype=target_dtype)
                    audio_embeds_input = audio_embeds.to(dtype=target_dtype)
                    
                    # UNet forward pass
                    noise_pred = unet_model(
                        sample=unet_input,
                        timestep=t,
                        encoder_hidden_states=audio_embeds_input
                    ).sample
                    
                    # 🔧 CRITICAL DEBUG: Check UNet output quality
                    if i == 0:  # Only debug first step to avoid spam
                        print(f"🔧 DEBUG - UNet output analysis (step 1):")
                        print(f"   - Shape: {noise_pred.shape}")
                        print(f"   - Dtype: {noise_pred.dtype}")
                        print(f"   - Mean: {noise_pred.mean():.6f}, Std: {noise_pred.std():.6f}")
                        print(f"   - Range: [{noise_pred.min():.6f}, {noise_pred.max():.6f}]")
                        
                        # Check if UNet is producing meaningful predictions
                        if torch.all(noise_pred == 0):
                            print("⚠️ CRITICAL: UNet output is all zeros!")
                            print("🔧 This indicates a model loading or processing issue")
                        elif noise_pred.std() < 0.001:
                            print("⚠️ WARNING: UNet output has very low variance!")
                            print("🔧 This may indicate poor audio conditioning")
                        else:
                            print("✅ UNet output appears to have meaningful content")
                    
                    # Apply classifier-free guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
                        # 🔧 CRITICAL DEBUG: Check CFG effect
                        if i == 0:  # Only debug first step
                            print(f"🔧 DEBUG - CFG effect analysis:")
                            print(f"   - Unconditional mean: {noise_pred_uncond.mean():.6f}")
                            print(f"   - Conditional mean: {noise_pred_text.mean():.6f}")
                            print(f"   - Difference: {(noise_pred_text - noise_pred_uncond).mean():.6f}")
                            print(f"   - Final guided mean: {noise_pred.mean():.6f}")
                            
                            cfg_effect = (noise_pred_text - noise_pred_uncond).abs().mean()
                            print(f"   - CFG effect magnitude: {cfg_effect:.6f}")
                            
                            if cfg_effect < 0.001:
                                print("⚠️ WARNING: CFG has minimal effect!")
                                print("🔧 Audio conditioning may not be working properly")
                            else:
                                print("✅ CFG is having meaningful effect")
                    
                    # Scheduler step (official denoising)
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                    
                    # 🚀 PERFORMANCE: Ensure latents maintain target dtype
                    latents = latents.to(dtype=target_dtype)
                    
                    # 🧹 OPTIMIZED CLEANUP: Less frequent cache clearing
                    if i % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                print("✅ Denoising complete!")
                log_memory_usage("🔄 Denoising complete")
                
                # 12. Decode latents back to image
                print("🎨 Decoding latents to image...")
                latents = latents.squeeze(2)  # Remove frame dimension
                
                # 🔧 CRITICAL FIX: Proper VAE decoding for LatentSync
                # Use the correct scaling factor without shift (LatentSync uses standard VAE)
                latents = latents / vae.config.scaling_factor
                
                decoded_image = vae.decode(latents).sample
                
                # 🔧 CRITICAL DEBUG: Check VAE output
                print(f"🔧 DEBUG - VAE output: shape={decoded_image.shape}, dtype={decoded_image.dtype}, range=[{decoded_image.min():.3f}, {decoded_image.max():.3f}]")
                
                # 🔧 CRITICAL FIX: Check for resolution mismatch
                vae_resolution = decoded_image.shape[-1]  # Get width from VAE output
                expected_resolution = 512  # FaceFusion crop frame resolution
                
                if vae_resolution != expected_resolution:
                    print(f"⚠️ RESOLUTION MISMATCH DETECTED!")
                    print(f"   - VAE output: {vae_resolution}x{vae_resolution}")
                    print(f"   - Expected: {expected_resolution}x{expected_resolution}")
                    print(f"   - This will cause lip size/quality issues when resized")
                    print(f"🔧 Consider using 'quality' performance mode for native 512x512 processing")
                else:
                    print(f"✅ Resolution match: {vae_resolution}x{vae_resolution} (optimal)")
                
                # 🔧 CRITICAL FIX: Proper color space normalization
                # Convert from [-1, 1] to [0, 1] range properly
                decoded_image = (decoded_image + 1.0) / 2.0
                decoded_image = decoded_image.clamp(0, 1)
                
                print(f"🔧 DEBUG - After normalization: range=[{decoded_image.min():.3f}, {decoded_image.max():.3f}]")
                
                # Convert to numpy and ensure proper channel order
                decoded_image = decoded_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                decoded_image = (decoded_image * 255).astype(numpy.uint8)
                
                print(f"🔧 DEBUG - After numpy conversion: shape={decoded_image.shape}, dtype={decoded_image.dtype}, range=[{decoded_image.min()}, {decoded_image.max()}]")
                print(f"🔧 DEBUG - Channel analysis: R={decoded_image[:,:,0].mean():.1f}, G={decoded_image[:,:,1].mean():.1f}, B={decoded_image[:,:,2].mean():.1f}")
                
                # 🔧 CRITICAL DEBUG: Compare original vs processed in mouth region
                target_frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
                target_resized = cv2.resize(target_frame_rgb, (decoded_image.shape[1], decoded_image.shape[0]))
                
                # Calculate difference in mouth region (approximate)
                mouth_y_start = int(decoded_image.shape[0] * 0.6)
                mouth_y_end = int(decoded_image.shape[0] * 0.9)
                mouth_x_start = int(decoded_image.shape[1] * 0.3)
                mouth_x_end = int(decoded_image.shape[1] * 0.7)
                
                original_mouth = target_resized[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
                processed_mouth = decoded_image[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
                
                mouth_diff = numpy.abs(processed_mouth.astype(float) - original_mouth.astype(float)).mean()
                print(f"🔧 DEBUG - Mouth region analysis:")
                print(f"   - Original mouth mean: {original_mouth.mean():.1f}")
                print(f"   - Processed mouth mean: {processed_mouth.mean():.1f}")
                print(f"   - Mouth region difference: {mouth_diff:.1f}")
                
                if mouth_diff < 5.0:
                    print("⚠️ CRITICAL: Very small difference in mouth region!")
                    print("🔧 LatentSync may not be generating lip movement")
                    print("🔧 Possible causes:")
                    print("   - Silent or low-quality audio")
                    print("   - Audio embeddings are zeros/constant")
                    print("   - UNet not responding to audio conditioning")
                    print("   - Model weights not loaded correctly")
                elif mouth_diff > 50.0:
                    print("⚠️ WARNING: Very large difference in mouth region!")
                    print("🔧 This may indicate color space issues or over-processing")
                else:
                    print("✅ Reasonable difference in mouth region - lip sync may be working")
                
                # 🔧 CRITICAL FIX: Check for color channel issues in VAE output
                r_mean = decoded_image[:,:,0].mean()
                g_mean = decoded_image[:,:,1].mean()
                b_mean = decoded_image[:,:,2].mean()
                
                # If red channel is significantly higher, there might be a color space issue
                if r_mean > g_mean * 1.3 and r_mean > b_mean * 1.3:
                    print("⚠️ DETECTED: VAE output has red channel dominance")
                    print("🔧 Applying color correction at VAE level...")
                    
                    # Try RGB to BGR conversion
                    decoded_image_corrected = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2BGR)
                    print(f"🔧 After RGB->BGR: R={decoded_image_corrected[:,:,0].mean():.1f}, G={decoded_image_corrected[:,:,1].mean():.1f}, B={decoded_image_corrected[:,:,2].mean():.1f}")
                    
                    # Check if this looks more balanced
                    r_corrected = decoded_image_corrected[:,:,0].mean()
                    g_corrected = decoded_image_corrected[:,:,1].mean()
                    b_corrected = decoded_image_corrected[:,:,2].mean()
                    
                    # If the corrected version is more balanced, use it
                    if abs(r_corrected - g_corrected) < abs(r_mean - g_mean):
                        print("✅ Color correction improved balance, using corrected version")
                        decoded_image = decoded_image_corrected
                    else:
                        print("⚠️ Color correction didn't help, keeping original")
                
                # 🔧 CRITICAL FIX: Ensure RGB format before conversion
                # The decoded image should already be in RGB format from VAE
                
                # 14. Resize back to original frame size
                original_height, original_width = target_frame.shape[:2]
                
                # 🔧 CRITICAL FIX: Ensure we're working with RGB data for PIL
                if decoded_image.shape[2] == 3:
                    # Check if this is already BGR (OpenCV format) or RGB (PIL format)
                    # If we applied BGR correction above, we need to convert back to RGB for PIL
                    if r_mean > g_mean * 1.3 and r_mean > b_mean * 1.3:
                        # We likely corrected to BGR, so convert back to RGB for PIL
                        decoded_image_for_pil = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
                        print("🔧 Converting BGR back to RGB for PIL processing")
                    else:
                        decoded_image_for_pil = decoded_image
                        print("🔧 Using original RGB for PIL processing")
                else:
                    decoded_image_for_pil = decoded_image
                
                decoded_pil = Image.fromarray(decoded_image_for_pil, mode='RGB')  # Explicitly specify RGB
                decoded_pil = decoded_pil.resize((original_width, original_height), Image.LANCZOS)
                result_frame = numpy.array(decoded_pil)
                
                print(f"🔧 DEBUG - After PIL resize: shape={result_frame.shape}, dtype={result_frame.dtype}, range=[{result_frame.min()}, {result_frame.max()}]")
                print(f"🔧 DEBUG - PIL output channels: R={result_frame[:,:,0].mean():.1f}, G={result_frame[:,:,1].mean():.1f}, B={result_frame[:,:,2].mean():.1f}")
                
                # 🔧 CRITICAL FIX: Convert RGB to BGR for OpenCV compatibility
                result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                
                print(f"🔧 DEBUG - Final result (BGR): shape={result_frame.shape}, dtype={result_frame.dtype}, range=[{result_frame.min()}, {result_frame.max()}]")
                print(f"🔧 DEBUG - Final BGR channels: B={result_frame[:,:,0].mean():.1f}, G={result_frame[:,:,1].mean():.1f}, R={result_frame[:,:,2].mean():.1f}")
                
                log_memory_usage("🎨 Frame processing complete")
                
                return result_frame
            
            # 🧹 CLEAN SCOPE EXIT: torch.no_grad() context automatically cleans up
            # Only need minimal cleanup for CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        else:
            # ===== FALLBACK TO ONNX WAV2LIP =====
            return process_frame_wav2lip(source_face, target_frame, audio_chunk)
            
    except Exception as error:
        print(f"❌ Error in process_frame_latentsync: {error}")
        import traceback
        traceback.print_exc()
        
        # 🧹 SIMPLE ERROR CLEANUP: Just clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Return original frame on error
        return target_frame


def process_frame(inputs : LipSyncerInputs) -> VisionFrame:
    """
    Standard FaceFusion interface for process_frame - takes dictionary input
    """
    # Extract inputs from dictionary
    reference_faces = inputs.get('reference_faces')
    source_audio_frame = inputs.get('source_audio_frame')
    target_vision_frame = inputs.get('target_vision_frame')
    
    # Call the original process_frame_original function
    return process_frame_original(inputs)


def process_frames(source_paths : List[str], queue_payloads : List[QueuePayload], update_progress : UpdateProgress) -> None:
	reference_faces = get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else None
	source_audio_path = get_first(filter_audio_paths(source_paths))
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
	model_name = state_manager.get_item('lip_syncer_model')

	# 🔧 DEBUG: Add debug output to see what's happening
	print(f"🔍 DEBUG process_frames:")
	print(f"   - source_paths: {source_paths}")
	print(f"   - source_audio_path: {source_audio_path}")
	print(f"   - model_name: {model_name}")
	print(f"   - total_frames: {len(queue_payloads)}")

	# 🚀 OFFICIAL LATENTSYNC: Use 2-step audio processing
	audio_chunks = None
	if model_name == 'latentsync' and source_audio_path:
		print("🎵 Using OFFICIAL LatentSync 2-step audio processing...")
		total_frames = len(queue_payloads)
		
		try:
			# 🔧 OFFICIAL METHOD: Create audio chunks using feature2chunks workflow
			audio_chunks = create_audio_chunks_official(source_audio_path, temp_video_fps, total_frames)
			print(f"✅ Official LatentSync audio processing complete!")
			print(f"   - Pre-computed features once (efficient)")
			print(f"   - Created {len(audio_chunks)} small audio slices")
			print(f"   - Memory per slice: ~{audio_chunks[0].numel() * 4 / 1024:.1f} KB (TINY!)")
			print(f"   - Total video frames to process: {total_frames}")
			print(f"   - Audio chunks available: {len(audio_chunks)}")
		except Exception as e:
			print(f"❌ Official audio processing failed: {e}")
			print(f"🔧 Will fall back to raw audio method per frame")
			audio_chunks = None
	else:
		print(f"🔧 Not using official LatentSync audio processing:")
		print(f"   - model_name == 'latentsync': {model_name == 'latentsync'}")
		print(f"   - source_audio_path exists: {source_audio_path is not None}")

	# 🔧 CRITICAL FIX: Create frame number to index mapping
	frame_to_index = {}
	for idx, queue_payload in enumerate(queue_payloads):
		frame_number = queue_payload.get('frame_number')
		frame_to_index[frame_number] = idx

	for queue_payload in process_manager.manage(queue_payloads):
		frame_number = queue_payload.get('frame_number')
		target_vision_path = queue_payload.get('frame_path')
		
		# 🔧 CRITICAL FIX: Get appropriate audio frame based on model
		if model_name == 'latentsync':
			if audio_chunks is not None:
				# 🔧 CRITICAL FIX: Use frame index instead of frame number
				frame_index = frame_to_index.get(frame_number, 0)
				
				if frame_index < len(audio_chunks):
					source_audio_frame = audio_chunks[frame_index]
					# Only log occasionally to avoid spam
					if frame_index % 100 == 0 or frame_index < 5:
						print(f"🔍 Frame {frame_number} (index {frame_index}): Using official audio slice {source_audio_frame.shape}")
				else:
					# Create small dummy slice if beyond audio length
					source_audio_frame = torch.zeros(10, 384)  # Official LatentSync slice size
					print(f"🔍 Frame {frame_number} (index {frame_index}): Using dummy audio slice (beyond audio length)")
			else:
				# 🔧 CRITICAL FIX: Use raw audio instead of mel-spectrogram for LatentSync
				print(f"⚠️ Using raw audio fallback for LatentSync frame {frame_number}")
				
				# Get raw audio chunk for this frame
				source_audio_frame = get_raw_audio_frame(source_audio_path, temp_video_fps, frame_number)
				
				if source_audio_frame is None:
					print(f"❌ Failed to get raw audio for frame {frame_number}")
					# Create minimal dummy audio
					source_audio_frame = numpy.zeros(16000, dtype=numpy.float32)  # 1 second of silence
				else:
					print(f"✅ Got raw audio for frame {frame_number}: shape {source_audio_frame.shape}, RMS {numpy.sqrt(numpy.mean(source_audio_frame**2)):.6f}")
		
		# 🔧 CRITICAL: Ensure we never pass None or empty audio to LatentSync
		if model_name == 'latentsync' and source_audio_frame is None:
			source_audio_frame = numpy.zeros(16000, dtype=numpy.float32)
		elif model_name != 'latentsync':
			# Traditional Wav2Lip: Use mel-spectrogram processing
			if source_audio_frame is None:
				source_audio_frame = get_voice_frame(source_audio_path, temp_video_fps, frame_number)
			if not numpy.any(source_audio_frame):
				source_audio_frame = create_empty_audio_frame()
		
		target_vision_frame = read_image(target_vision_path)
		output_vision_frame = process_frame_original(
		{
			'reference_faces': reference_faces,
			'source_audio_frame': source_audio_frame,
			'target_vision_frame': target_vision_frame
		})
		write_image(target_vision_path, output_vision_frame)
		update_progress(1)


def process_image(source_paths : List[str], target_path : str, output_path : str) -> None:
	# 🔧 DEBUG: Add debug output to see if process_image is being called
	print(f"🔍 DEBUG process_image:")
	print(f"   - source_paths: {source_paths}")
	print(f"   - target_path: {target_path}")
	print(f"   - output_path: {output_path}")
	print(f"   - model: {state_manager.get_item('lip_syncer_model')}")

	# 🔧 DEBUG: Check file type detection
	print(f"🔍 DEBUG file type detection:")
	print(f"   - is_image(target_path): {is_image(target_path)}")
	print(f"   - is_video(target_path): {is_video(target_path)}")
	print(f"   - target_path exists: {os.path.exists(target_path)}")

	reference_faces = get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else None
	model_name = state_manager.get_item('lip_syncer_model')
	
	# Create appropriate empty audio frame based on model
	if model_name == 'latentsync':
		# Create small empty audio slice for LatentSync (official format)
		source_audio_frame = torch.zeros(10, 384)  # Official LatentSync slice size
		print(f"🔍 DEBUG: Created LatentSync audio slice: {source_audio_frame.shape}")
	else:
		source_audio_frame = create_empty_audio_frame()
		print(f"🔍 DEBUG: Created traditional audio frame")
	
	target_vision_frame = read_static_image(target_path)
	output_vision_frame = process_frame_original(
	{
		'reference_faces': reference_faces,
		'source_audio_frame': source_audio_frame,
		'target_vision_frame': target_vision_frame
	})
	write_image(output_path, output_vision_frame)


def process_video(source_paths : List[str], temp_frame_paths : List[str]) -> None:
	# 🔧 DEBUG: Add debug output to see if process_video is being called
	print(f"🔍 DEBUG process_video:")
	print(f"   - source_paths: {source_paths}")
	print(f"   - temp_frame_paths count: {len(temp_frame_paths) if temp_frame_paths else 0}")
	print(f"   - model: {state_manager.get_item('lip_syncer_model')}")
	
	# 🎵 Pre-load audio for traditional models (Wav2Lip compatibility)
	source_audio_paths = filter_audio_paths(state_manager.get_item('source_paths'))
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
	for source_audio_path in source_audio_paths:
		read_static_voice(source_audio_path, temp_video_fps)
	
	# 🚀 Note: LatentSync now uses batch audio processing in process_frames()
	# This improves temporal consistency and matches official LatentSync behavior
	print(f"🔍 DEBUG: About to call multi_process_frames with process_frames function")
	processors.multi_process_frames(source_paths, temp_frame_paths, process_frames)

def get_unet_model():
    """Lazy loading of PyTorch UNet model"""
    global unet_model
    if unet_model is None:
        if not LATENTSYNC_AVAILABLE:
            raise RuntimeError("LatentSync is not available. Cannot load UNet model.")
            
        print("🧠 Loading PyTorch UNet model...")
        
        # Define paths
        config_path = "/home/cody_braiv_co/latent-sync/configs/unet/stage2.yaml"
        ckpt_path = "/home/cody_braiv_co/latent-sync/checkpoints/latentsync_unet.pt"
        
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Loading UNet on {target_device}")
        
        try:
            # Load config and initialize model
            config = OmegaConf.load(config_path)
            unet_model = UNet3DConditionModel(**config.model)
            
            # Load checkpoint weights
            checkpoint = torch.load(ckpt_path, map_location=target_device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            unet_model.load_state_dict(state_dict)
            
            # 🔧 CRITICAL FIX: Set UNet precision based on performance mode
            perf_config = get_performance_config()
            if perf_config["use_fp16"] and torch.cuda.is_available():
                unet_model = unet_model.half().to(target_device).eval()
                print(f"🚀 UNet converted to FP16 for performance consistency")
            else:
                unet_model = unet_model.float().to(target_device).eval()
                print(f"🔧 UNet kept in FP32 precision")
            
            # Clean up checkpoint from memory
            del checkpoint, state_dict
            torch.cuda.empty_cache()
            
            print(f"✅ PyTorch UNet loaded on {target_device} with {'FP16' if unet_model.dtype == torch.float16 else 'FP32'} precision")
            
        except Exception as load_error:
            print(f"❌ Failed to load PyTorch UNet: {load_error}")
            unet_model = None
            raise RuntimeError(f"Could not initialize PyTorch UNet: {load_error}")
    else:
        # 🔧 CRITICAL: Ensure UNet precision matches performance mode
        perf_config = get_performance_config()
        target_dtype = torch.float16 if perf_config["use_fp16"] and torch.cuda.is_available() else torch.float32
        
        if unet_model.dtype != target_dtype:
            print(f"🔧 Converting UNet from {unet_model.dtype} to {target_dtype} for consistency")
            if target_dtype == torch.float16:
                unet_model = unet_model.half()
            else:
                unet_model = unet_model.float()
    
    return unet_model

def get_scheduler():
    """Lazy loading of DDIM scheduler"""
    global scheduler
    if scheduler is None:
        print("📅 Loading DDIM scheduler...")
        # Use default DDIM scheduler with official LatentSync parameters
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon"
        )
        print("✅ DDIM scheduler loaded with official LatentSync parameters")
    
    return scheduler

def prepare_latents(batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator=None):
    """Prepare initial latents for denoising (matching official LatentSync)"""
    shape = (batch_size, num_channels_latents, num_frames, height, width)
    
    # Initialize with random noise in the correct dtype
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    
    # Scale by scheduler's init noise sigma
    scheduler = get_scheduler()
    latents = latents * scheduler.init_noise_sigma
    
    return latents

def process_frame_wav2lip(source_face: Face, target_frame: VisionFrame, audio_chunk: numpy.ndarray) -> VisionFrame:
    """Fallback to original ONNX Wav2Lip processing"""
    # This is a placeholder - you can implement the original ONNX logic here
    # For now, just return the original frame
    print("⚠️ Falling back to original frame (Wav2Lip not implemented)")
    return target_frame

def create_latentsync_mouth_mask_from_landmarks(target_frame: VisionFrame, landmarks_68: numpy.ndarray) -> numpy.ndarray:
    """
    Create a mouth region mask using transformed landmarks directly.
    This avoids the Face() constructor issue while maintaining the same mask quality.
    """
    height, width = target_frame.shape[:2]
    
    if landmarks_68 is None or len(landmarks_68) < 68:
        print("⚠️ No valid landmarks for mouth mask - using fallback")
        # Fallback: create a simple rectangular mask in the lower face region
        mouth_mask = numpy.zeros((height, width), dtype=numpy.float32)
        mouth_y_start = int(height * 0.6)
        mouth_y_end = int(height * 0.9)
        mouth_x_start = int(width * 0.3)
        mouth_x_end = int(width * 0.7)
        mouth_mask[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end] = 1.0
        return mouth_mask
    
    # Create mouth mask using landmarks (same logic as original function)
    mouth_mask = numpy.zeros((height, width), dtype=numpy.float32)
    
    # Get mouth landmarks (indices 48-67 in 68-point landmarks)
    mouth_landmarks = landmarks_68[48:68].astype(numpy.int32)
    
    # 🔧 CRITICAL DEBUG: Check landmark positions
    print(f"🔧 DEBUG - Mouth landmarks range: x=[{mouth_landmarks[:, 0].min()}, {mouth_landmarks[:, 0].max()}], y=[{mouth_landmarks[:, 1].min()}, {mouth_landmarks[:, 1].max()}]")
    
    # Create convex hull around mouth region
    hull = cv2.convexHull(mouth_landmarks)
    cv2.fillPoly(mouth_mask, [hull], 1.0)
    
    # 🔧 CRITICAL DEBUG: Check mask after fillPoly
    initial_coverage = numpy.sum(mouth_mask > 0) / mouth_mask.size
    print(f"🔧 DEBUG - Initial mask coverage: {initial_coverage:.1%}")
    
    # 🔧 CRITICAL FIX: Use proper erosion kernel (must be odd-sized and uint8)
    erosion_kernel = numpy.ones((3, 3), numpy.uint8)  # Much smaller kernel to preserve mask
    mouth_mask = cv2.erode(mouth_mask, erosion_kernel, iterations=1)
    
    # 🔧 CRITICAL DEBUG: Check mask after erosion
    eroded_coverage = numpy.sum(mouth_mask > 0) / mouth_mask.size
    print(f"🔧 DEBUG - After erosion coverage: {eroded_coverage:.1%}")
    
    # Apply Gaussian blur for smooth edges
    mouth_mask = cv2.GaussianBlur(mouth_mask, (15, 15), 0)
    
    # 🔧 CRITICAL DEBUG: Check final mask
    final_coverage = numpy.sum(mouth_mask > 0.01) / mouth_mask.size
    print(f"🔧 DEBUG - Final mask coverage: {final_coverage:.1%}")
    
    # 🔧 CRITICAL FIX: Resize to target frame size if needed
    if (height, width) != (512, 512):
        mouth_mask = cv2.resize(mouth_mask, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return mouth_mask

def create_latentsync_mouth_mask(target_frame: VisionFrame, target_face: Face) -> numpy.ndarray:
    """Create a mouth region mask for inpainting using FaceFusion's high-quality method"""
    height, width = target_frame.shape[:2]
    
    # 🔧 CRITICAL FIX: Use the transformed landmarks that match the 512x512 crop
    # The target_face landmarks are in original image coordinates, but we need crop coordinates
    # This function should receive the transformed landmarks instead
    
    # Get face landmarks if available
    if target_face is not None and hasattr(target_face, 'landmark_set') and target_face.landmark_set is not None:
        landmarks_68 = target_face.landmark_set.get('68')
        
        if landmarks_68 is not None and len(landmarks_68) >= 68:
            # 🔧 CRITICAL FIX: Use FaceFusion's high-quality mouth mask method
            # This matches the create_mouth_mask function from face_masker.py
            
            # Create convex hull around mouth region (same as FaceFusion)
            convex_hull = cv2.convexHull(landmarks_68[numpy.r_[3:14, 31:36]].astype(numpy.int32))
            
            # Create mask at current frame resolution
            mouth_mask = numpy.zeros((height, width)).astype(numpy.float32)
            mouth_mask = cv2.fillConvexPoly(mouth_mask, convex_hull, 1.0)
            
            # Apply erosion and Gaussian blur (same as FaceFusion)
            mouth_mask = cv2.erode(mouth_mask.clip(0, 1), numpy.ones((21, 3)))
            mouth_mask = cv2.GaussianBlur(mouth_mask, (0, 0), sigmaX=1, sigmaY=15)
            
            return mouth_mask
    
    # Fallback: create a simple rectangular mask in the lower face region
    mask = numpy.zeros((height, width), dtype=numpy.float32)
    center_x, center_y = width // 2, int(height * 0.7)
    mask_width, mask_height = width // 4, height // 6
    
    x1 = max(0, center_x - mask_width // 2)
    x2 = min(width, center_x + mask_width // 2)
    y1 = max(0, center_y - mask_height // 2)
    y2 = min(height, center_y + mask_height // 2)
    
    mask[y1:y2, x1:x2] = 1.0
    
    # Apply some blur to the fallback mask too
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5, sigmaY=5)
    
    return mask

def process_frame_original(inputs : LipSyncerInputs) -> VisionFrame:
    """Original process_frame function that wraps the new implementation"""
    global audio_encoder, vae  # Move global declarations to top
    
    reference_faces = inputs.get('reference_faces')
    source_audio_frame = inputs.get('source_audio_frame')
    target_vision_frame = inputs.get('target_vision_frame')
    many_faces = sort_and_filter_faces(get_many_faces([ target_vision_frame ]))

    if state_manager.get_item('face_selector_mode') == 'many':
        if many_faces:
            for target_face in many_faces:
                target_vision_frame = sync_lip(target_face, source_audio_frame, target_vision_frame)
    if state_manager.get_item('face_selector_mode') == 'one':
        target_face = get_one_face(many_faces)
        if target_face:
            target_vision_frame = sync_lip(target_face, source_audio_frame, target_vision_frame)
    if state_manager.get_item('face_selector_mode') == 'reference':
        similar_faces = find_similar_faces(many_faces, reference_faces, state_manager.get_item('reference_face_distance'))
        if similar_faces:
            for similar_face in similar_faces:
                target_vision_frame = sync_lip(similar_face, source_audio_frame, target_vision_frame)
    
    # 🧹 AGGRESSIVE CLEANUP after each frame to prevent memory leaks
    model_name = state_manager.get_item('lip_syncer_model')
    if model_name == 'latentsync':
        # 🔥 Simple memory cleanup without device switching
        try:
            print("🧹 Cleaning up GPU memory after LatentSync frame...")
            
            # Simple GPU cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
                # Check memory after cleanup
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"💾 GPU memory after cleanup: {allocated:.2f} GB")
            
            print("✅ Memory cleanup completed")
            
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup warning: {cleanup_error}")
    
    return target_vision_frame


def sync_lip(target_face: Face, temp_audio_frame: AudioFrame, temp_vision_frame: VisionFrame) -> VisionFrame:
    model_size = get_model_options().get('size')
    model_name = state_manager.get_item('lip_syncer_model')
    
    # 🔧 CRITICAL FIX: Handle LatentSync audio differently from Wav2Lip
    if model_name == 'latentsync':
        # LatentSync uses raw audio or pre-computed audio slices, NOT mel-spectrograms
        print(f"🔧 LatentSync audio processing:")
        print(f"   - Input type: {type(temp_audio_frame)}")
        print(f"   - Input shape: {temp_audio_frame.shape if hasattr(temp_audio_frame, 'shape') else 'No shape'}")
        
        # Check if this is already a pre-computed audio slice (from official method)
        if isinstance(temp_audio_frame, torch.Tensor) and temp_audio_frame.dim() == 2 and temp_audio_frame.shape[1] == 384:
            print(f"✅ Using pre-computed LatentSync audio slice: {temp_audio_frame.shape}")
            audio_input = temp_audio_frame
        elif isinstance(temp_audio_frame, numpy.ndarray) and temp_audio_frame.ndim == 1:
            print(f"✅ Using raw audio waveform: {temp_audio_frame.shape}")
            audio_input = temp_audio_frame
        else:
            print(f"🚨 CRITICAL: Still receiving mel-spectrogram for LatentSync!")
            print(f"🔧 Shape {temp_audio_frame.shape} indicates mel-spectrogram format")
            print(f"🔧 LatentSync requires raw audio - extracting from source...")
            
            # 🔧 EMERGENCY FIX: Extract raw audio directly from source
            try:
                source_paths = state_manager.get_item('source_paths', [])
                source_audio_path = None
                
                for path in source_paths:
                    if has_audio(path):
                        source_audio_path = path
                        break
                
                if source_audio_path:
                    print(f"🔧 Extracting raw audio from: {source_audio_path}")
                    
                    # 🔧 DEBUG: Check if file exists and get basic info
                    if not os.path.exists(source_audio_path):
                        print(f"❌ Audio file does not exist: {source_audio_path}")
                        raise FileNotFoundError(f"Audio file not found: {source_audio_path}")
                    
                    file_size = os.path.getsize(source_audio_path)
                    print(f"🔧 Audio file info: size={file_size} bytes, exists={os.path.exists(source_audio_path)}")
                    
                    # Read raw audio at 16kHz for LatentSync
                    try:
                        audio_data, sample_rate = sf.read(source_audio_path)
                        print(f"🔧 Raw audio loaded: shape={audio_data.shape}, sample_rate={sample_rate}, dtype={audio_data.dtype}")
                        print(f"🔧 Raw audio stats: mean={audio_data.mean():.6f}, std={audio_data.std():.6f}, range=[{audio_data.min():.6f}, {audio_data.max():.6f}]")
                        
                        # Check if audio is actually silent
                        audio_rms = numpy.sqrt(numpy.mean(audio_data**2))
                        print(f"🔧 Raw audio RMS: {audio_rms:.6f}")
                        
                        if audio_rms < 0.001:
                            print(f"⚠️ WARNING: Source audio file appears to be silent!")
                            print(f"🔧 This may be the root cause of the lip sync issue")
                        
                    except Exception as read_error:
                        print(f"❌ Failed to read audio file: {read_error}")
                        raise read_error
                        
                    # Ensure mono
                    if len(audio_data.shape) > 1:
                        print(f"🔧 Converting from {audio_data.shape} to mono")
                        audio_data = numpy.mean(audio_data, axis=1)
                    
                    # Resample to 16kHz if needed
                    if sample_rate != 16000:
                        print(f"🔧 Resampling from {sample_rate}Hz to 16000Hz")
                        audio_data = resample_audio(audio_data, sample_rate, 16000)
                    
                    # Use a reasonable chunk size (1 second)
                    chunk_size = 16000
                    if len(audio_data) > chunk_size:
                        # Take first second of audio
                        print(f"🔧 Truncating audio from {len(audio_data)} to {chunk_size} samples")
                        audio_data = audio_data[:chunk_size]
                    elif len(audio_data) < chunk_size:
                        # Pad with zeros if too short
                        print(f"🔧 Padding audio from {len(audio_data)} to {chunk_size} samples")
                        audio_data = numpy.pad(audio_data, (0, chunk_size - len(audio_data)), mode='constant')
                    
                    audio_input = audio_data.astype(numpy.float32)
                    
                    print(f"✅ Successfully extracted raw audio:")
                    print(f"   - Shape: {audio_input.shape}")
                    print(f"   - Duration: {len(audio_input) / 16000:.2f} seconds")
                    print(f"   - RMS: {numpy.sqrt(numpy.mean(audio_input**2)):.6f}")
                    
                    # Verify audio is not silent
                    if numpy.sqrt(numpy.mean(audio_input**2)) < 0.001:
                        print(f"⚠️ WARNING: Extracted audio appears silent!")
                        print(f"🔧 This confirms the source audio file is silent or very quiet")
                        print(f"🔧 Lip sync will not work with silent audio")
                    else:
                        print(f"✅ Audio extraction successful - non-silent audio detected")
                        
                else:
                    raise RuntimeError("No audio source found in source_paths")
                    
            except Exception as audio_error:
                print(f"❌ Raw audio extraction failed: {audio_error}")
                print(f"🔧 Creating dummy audio - NO LIP SYNC will occur!")
                audio_input = numpy.zeros(16000, dtype=numpy.float32)  # 1 second of silence
    else:
        # Traditional Wav2Lip uses mel-spectrograms
        temp_audio_frame = prepare_audio_frame(temp_audio_frame)
        audio_input = temp_audio_frame
    
    crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark_set.get('5/68'), 'ffhq_512', (512, 512))
    face_landmark_68 = cv2.transform(target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
    bounding_box = create_bounding_box(face_landmark_68)
    bounding_box[1] -= numpy.abs(bounding_box[3] - bounding_box[1]) * 0.125
    
    # 🔧 CRITICAL FIX: Simplified LatentSync processing
    # LatentSync already implements the complete two-stage algorithm internally
    # No need to duplicate the logic here - just call forward() and paste back
    
    if model_name == 'latentsync':
        print("🔧 LatentSync: Using internal two-stage algorithm (no external processing needed)")
        
        # LatentSync handles everything internally - just call forward()
        processed_face = forward(audio_input, crop_vision_frame, target_face)
        
        # Check if processing was successful
        if processed_face is None or not isinstance(processed_face, numpy.ndarray):
            print("⚠️ LatentSync processing failed, using original frame")
            return temp_vision_frame
            
        if processed_face.shape != (512, 512, 3):
            print(f"⚠️ Unexpected LatentSync output shape: {processed_face.shape}")
            return temp_vision_frame
        
        # 🔧 DEBUG: Final image analysis - check if mouth region changed
        print(f"🔧 Decoded image - Shape: {processed_face.shape}, dtype: {processed_face.dtype}, range: [{processed_face.min():.1f}, {processed_face.max():.1f}]")
        
        # Convert target frame to RGB and resize to match decoded image
        target_rgb = cv2.cvtColor(crop_vision_frame, cv2.COLOR_BGR2RGB) if len(crop_vision_frame.shape) == 3 else crop_vision_frame
        target_resized = cv2.resize(target_rgb, (processed_face.shape[1], processed_face.shape[0]))
        
        # Check mouth region difference (approximate mouth area)
        mouth_y1, mouth_y2 = int(processed_face.shape[0] * 0.6), int(processed_face.shape[0] * 0.85)
        mouth_x1, mouth_x2 = int(processed_face.shape[1] * 0.3), int(processed_face.shape[1] * 0.7)
        
        mouth_orig = target_resized[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        mouth_proc = processed_face[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        mouth_diff = numpy.mean(numpy.abs(mouth_orig.astype(float) - mouth_proc.astype(float)))
        
        print(f"🔧 Mouth region difference: {mouth_diff:.2f}")
        if mouth_diff < 5.0:
            print("⚠️ Very small mouth difference - model may not be generating lip sync")
        elif mouth_diff > 50.0:
            print("⚠️ Very large mouth difference - may indicate processing issues")

        # Apply color correction if needed
        color_corrected = False
        if processed_face.dtype == numpy.uint8:
            # Check for color imbalance (common issue with LatentSync)
            r_mean = processed_face[:,:,0].mean()
            g_mean = processed_face[:,:,1].mean() 
            b_mean = processed_face[:,:,2].mean()
            
            print(f"🔧 Channel means - R: {r_mean:.1f}, G: {g_mean:.1f}, B: {b_mean:.1f}")
            
            # Detect significant color imbalance
            if abs(r_mean - g_mean) > 20 or abs(r_mean - b_mean) > 20 or abs(g_mean - b_mean) > 20:
                print("🔧 Applying color correction for channel imbalance...")
                
                # Calculate correction factors
                target_mean = (r_mean + g_mean + b_mean) / 3
                r_factor = target_mean / max(r_mean, 1)
                g_factor = target_mean / max(g_mean, 1)
                b_factor = target_mean / max(b_mean, 1)
                
                # Apply correction
                processed_face_corrected = processed_face.astype(numpy.float32)
                processed_face_corrected[:,:,0] *= r_factor
                processed_face_corrected[:,:,1] *= g_factor
                processed_face_corrected[:,:,2] *= b_factor
                processed_face_corrected = numpy.clip(processed_face_corrected, 0, 255).astype(numpy.uint8)
                
                print(f"🔧 After correction - Channel means: R={processed_face_corrected[:,:,0].mean():.1f}, G={processed_face_corrected[:,:,1].mean():.1f}, B={processed_face_corrected[:,:,2].mean():.1f}")
                processed_face = processed_face_corrected
                color_corrected = True
            
        if not color_corrected:
            print("✅ No color correction needed")
        
        # ===== LATENTSYNC TWO-STAGE ALGORITHM =====
        # Stage 1: Pixel-level masking (mouth region only)
        print("🔧 LatentSync Stage 1: Creating precise mouth mask...")
        
        # Create mouth mask using landmarks (much more precise than face mask)
        mouth_mask = create_latentsync_mouth_mask_from_landmarks(crop_vision_frame, face_landmark_68)
        
        # Check mouth mask coverage
        mouth_coverage = numpy.sum(mouth_mask > 0.1) / mouth_mask.size
        print(f"🔧 Mouth mask coverage: {mouth_coverage:.1%}")
        
        if mouth_coverage < 0.005:  # Less than 0.5% coverage
            print("⚠️ Mouth mask too small, creating fallback mouth region...")
            mouth_mask = numpy.zeros((512, 512), dtype=numpy.float32)
            # Create a small mouth region in the lower face
            center_x, center_y = 256, int(512 * 0.75)  # Lower face area
            mouth_width, mouth_height = 80, 40  # Small mouth region
            
            x1 = max(0, center_x - mouth_width // 2)
            x2 = min(512, center_x + mouth_width // 2)
            y1 = max(0, center_y - mouth_height // 2)
            y2 = min(512, center_y + mouth_height // 2)
            
            mouth_mask[y1:y2, x1:x2] = 1.0
            mouth_mask = cv2.GaussianBlur(mouth_mask, (15, 15), 0)
            mouth_coverage = numpy.sum(mouth_mask > 0.1) / mouth_mask.size
            print(f"🔧 Fallback mouth mask coverage: {mouth_coverage:.1%}")
        
        # Expand mouth mask to 3 channels for blending
        mouth_mask_3d = numpy.stack([mouth_mask, mouth_mask, mouth_mask], axis=2)
        
        # Stage 1: Blend only the mouth region
        # result = generated * mask + original * (1 - mask)
        print("🔧 LatentSync Stage 1: Blending mouth region...")
        stage1_result = (processed_face.astype(numpy.float32) * mouth_mask_3d + 
                        crop_vision_frame.astype(numpy.float32) * (1 - mouth_mask_3d))
        stage1_result = numpy.clip(stage1_result, 0, 255).astype(numpy.uint8)
        
        # Debug: Check blending result
        print(f"🔧 Stage 1 result - Channel means: R={stage1_result[:,:,0].mean():.1f}, G={stage1_result[:,:,1].mean():.1f}, B={stage1_result[:,:,2].mean():.1f}")
        
        # 🔧 DEBUG: Save debug images for visualization
        debug_dir = "/tmp/latentsync_debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        try:
            # Save original crop frame
            cv2.imwrite(f"{debug_dir}/01_original_crop.jpg", crop_vision_frame)
            
            # Save LatentSync output
            cv2.imwrite(f"{debug_dir}/02_latentsync_output.jpg", processed_face)
            
            # Save mouth mask visualization
            mouth_mask_vis = (mouth_mask * 255).astype(numpy.uint8)
            cv2.imwrite(f"{debug_dir}/03_mouth_mask.jpg", mouth_mask_vis)
            
            # Save mouth mask overlay on original
            mouth_overlay = crop_vision_frame.copy()
            mouth_overlay[:,:,2] = numpy.where(mouth_mask > 0.1, 255, mouth_overlay[:,:,2])  # Red overlay
            cv2.imwrite(f"{debug_dir}/04_mouth_mask_overlay.jpg", mouth_overlay)
            
            # Save Stage 1 result (mouth blended)
            cv2.imwrite(f"{debug_dir}/05_stage1_mouth_blended.jpg", stage1_result)
            
            print(f"🔧 Debug images saved to: {debug_dir}")
        except Exception as debug_error:
            print(f"⚠️ Debug image saving failed: {debug_error}")
        
        # Stage 2: Face restoration to original video (using broader face mask for seamless integration)
        print("🔧 LatentSync Stage 2: Creating face restoration mask...")
        
        # Create a broader face mask for seamless integration
        face_mask = numpy.zeros((512, 512), dtype=numpy.float32)
        
        if len(face_landmark_68) >= 17:
            # Use face contour landmarks (0-16) for face boundary
            face_contour = face_landmark_68[0:17]
            
            # Expand the face contour slightly for better blending
            center_x = face_contour[:, 0].mean()
            center_y = face_contour[:, 1].mean()
            
            # Expand contour by 10% for smoother blending
            expanded_contour = face_contour.copy().astype(numpy.float32)
            expanded_contour[:, 0] = center_x + (expanded_contour[:, 0] - center_x) * 1.1
            expanded_contour[:, 1] = center_y + (expanded_contour[:, 1] - center_y) * 1.1
            expanded_contour = numpy.clip(expanded_contour, 0, 511).astype(numpy.int32)
            
            # Create convex hull around expanded face contour
            hull = cv2.convexHull(expanded_contour)
            cv2.fillPoly(face_mask, [hull], 1.0)
            
            # Apply moderate erosion to avoid edge artifacts
            kernel = numpy.ones((7, 7), numpy.uint8)
            face_mask = cv2.erode(face_mask, kernel, iterations=1)
            
            # Apply Gaussian blur for smooth blending
            face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
            
            face_coverage = numpy.sum(face_mask > 0.1) / face_mask.size
            print(f"🔧 Face restoration mask coverage: {face_coverage:.1%}")
            
        else:
            print("⚠️ Using elliptical face mask for restoration...")
            center_x, center_y = 256, 256
            cv2.ellipse(face_mask, (center_x, center_y), (180, 220), 0, 0, 360, 1.0, -1)
            face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
            face_coverage = numpy.sum(face_mask > 0.1) / face_mask.size
            print(f"🔧 Elliptical face mask coverage: {face_coverage:.1%}")
        
        print(f"🔧 LatentSync: Using mouth-only blending (mouth coverage: {mouth_coverage:.1%})")
        
        # 🔧 DEBUG: Save final debug images
        try:
            # Save face restoration mask
            face_mask_vis = (face_mask * 255).astype(numpy.uint8)
            cv2.imwrite(f"{debug_dir}/06_face_restoration_mask.jpg", face_mask_vis)
            
            # Save face mask overlay
            face_overlay = stage1_result.copy()
            face_overlay[:,:,1] = numpy.where(face_mask > 0.1, 255, face_overlay[:,:,1])  # Green overlay
            cv2.imwrite(f"{debug_dir}/07_face_mask_overlay.jpg", face_overlay)
            
            print(f"🔧 Final debug images saved. Check {debug_dir} for visualization.")
        except Exception as debug_error:
            print(f"⚠️ Final debug image saving failed: {debug_error}")
        
        # Use the Stage 1 result (mouth-blended face) for paste-back
        return paste_back(temp_vision_frame, stage1_result, face_mask, affine_matrix)
    
    else:
        # Traditional Wav2Lip processing with masks and close matrix
        mouth_mask = create_mouth_mask(face_landmark_68)
        crop_masks = [mouth_mask]

        box_mask = create_static_box_mask(crop_vision_frame.shape[:2][::-1], 
                                        state_manager.get_item('face_mask_blur'), 
                                        state_manager.get_item('face_mask_padding'))
        crop_masks.append(box_mask)

        if 'occlusion' in state_manager.get_item('face_mask_types'):
            occlusion_mask = create_occlusion_mask(crop_vision_frame)
            crop_masks.append(occlusion_mask)
        
        # Wav2Lip processing
        close_vision_frame, close_matrix = warp_face_by_bounding_box(crop_vision_frame, bounding_box, model_size)
        close_vision_frame = prepare_crop_frame(close_vision_frame)
        close_vision_frame = forward(audio_input, close_vision_frame, target_face)
        close_vision_frame = normalize_close_frame(close_vision_frame)
        
        # Check if the model returned a valid frame
        if close_vision_frame is None:
            print("⚠️ Model returned None frame, using original frame")
            return temp_vision_frame

        if not isinstance(close_vision_frame, numpy.ndarray):
            print("⚠️ Invalid frame type:", type(close_vision_frame))
            return temp_vision_frame

        if close_vision_frame.shape != (96, 96, 3):
            print(f"⚠️ Unexpected Wav2Lip output shape: {close_vision_frame.shape}")
            return temp_vision_frame
        
        # Traditional paste-back for Wav2Lip
        crop_vision_frame = cv2.warpAffine(close_vision_frame, cv2.invertAffineTransform(close_matrix), (512, 512), borderMode=cv2.BORDER_REPLICATE)
        crop_mask = numpy.minimum.reduce(crop_masks)
        return paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)

def encode_audio_for_latentsync(audio_input):
    """
    Encode audio for LatentSync using Whisper
    Handles both raw audio arrays and pre-computed features
    🔧 CRITICAL FIX: Proper handling of short audio to prevent Whisper padding errors
    """
    target_device = get_device_for_lip_syncer()
    
    # If it's already a tensor with the right shape, use it directly
    if isinstance(audio_input, torch.Tensor):
        if audio_input.dim() == 3 and audio_input.shape[-1] == 384:
            print(f"🔍 Using pre-computed audio slice: {audio_input.shape}")
            return audio_input.to(target_device)
        elif audio_input.dim() == 2 and audio_input.shape[-1] == 384:
            print(f"🔍 Using pre-computed audio slice (adding batch dim): {audio_input.shape}")
            return audio_input.unsqueeze(0).to(target_device)
    
    # Handle raw audio input
    try:
        audio_encoder = get_audio_encoder()
        
        # Convert to numpy if needed
        if isinstance(audio_input, torch.Tensor):
            audio_numpy = audio_input.cpu().numpy()
        else:
            audio_numpy = numpy.array(audio_input, dtype=numpy.float32)
        
        # 🔧 CRITICAL FIX: Ensure audio is 1D
        if audio_numpy.ndim > 1:
            audio_numpy = audio_numpy.flatten()
        
        # 🔧 CRITICAL FIX: Whisper requires minimum 30-second chunks (480,000 samples at 16kHz)
        # But for memory efficiency, we'll use a smaller minimum that still works
        min_samples = 16000  # 1 second minimum - much more reasonable than 30 seconds
        max_samples = 16000 * 10  # 10 seconds max to prevent memory issues
        
        print(f"🔧 Input audio length: {len(audio_numpy)} samples")
        
        # Pad if too short
        if len(audio_numpy) < min_samples:
            print(f"⚠️ Audio too short ({len(audio_numpy)} samples), padding to {min_samples} samples")
            padding_needed = min_samples - len(audio_numpy)
            # 🔧 CRITICAL FIX: Use reflection padding for more natural audio extension
            if len(audio_numpy) > 0:
                # Repeat the audio to reach minimum length
                repeat_count = (min_samples // len(audio_numpy)) + 1
                audio_numpy = numpy.tile(audio_numpy, repeat_count)[:min_samples]
            else:
                # If completely empty, create minimal noise
                audio_numpy = numpy.random.normal(0, 0.001, min_samples).astype(numpy.float32)
        
        # Truncate if too long
        if len(audio_numpy) > max_samples:
            print(f"⚠️ Audio too long ({len(audio_numpy)} samples), truncating to {max_samples} samples")
            audio_numpy = audio_numpy[:max_samples]
        
        # 🔧 CRITICAL FIX: Ensure audio length is compatible with Whisper's requirements
        # Whisper works best with multiples of 160 samples (10ms frames at 16kHz)
        frame_size = 160
        if len(audio_numpy) % frame_size != 0:
            padding_needed = frame_size - (len(audio_numpy) % frame_size)
            audio_numpy = numpy.pad(audio_numpy, (0, padding_needed), mode='constant', constant_values=0)
            print(f"🔧 Aligned audio to frame boundary: {len(audio_numpy)} samples")
        
        print(f"🔧 Processed audio length: {len(audio_numpy)} samples")
        
        # Create temporary file for audio2feat
        temp_path = f"/tmp/temp_audio_{os.getpid()}_{int(time.time() * 1000)}.wav"
        
        try:
            # 🔧 CRITICAL FIX: Ensure proper audio format for Whisper
            if audio_numpy.dtype != numpy.float32:
                audio_numpy = audio_numpy.astype(numpy.float32)
            
            # 🔧 CRITICAL FIX: Normalize audio to [-1, 1] range
            if numpy.max(numpy.abs(audio_numpy)) > 0:
                audio_numpy = audio_numpy / numpy.max(numpy.abs(audio_numpy))
            
            # 🔧 CRITICAL FIX: Ensure audio is not all zeros (causes Whisper issues)
            if numpy.all(audio_numpy == 0):
                print("⚠️ Audio is all zeros, adding minimal noise for Whisper compatibility")
                audio_numpy = numpy.random.normal(0, 0.001, len(audio_numpy)).astype(numpy.float32)
            
            print(f"🔧 Processing audio: {len(audio_numpy)} samples, dtype: {audio_numpy.dtype}")
            print(f"🔧 Audio range: [{numpy.min(audio_numpy):.6f}, {numpy.max(audio_numpy):.6f}]")
            
            # Write audio file with proper format
            sf.write(temp_path, audio_numpy, 16000, format='WAV', subtype='PCM_16')
            
            # Verify file was written correctly
            if not os.path.exists(temp_path):
                raise RuntimeError("Failed to write temporary audio file")
            
            file_size = os.path.getsize(temp_path)
            print(f"🔧 Temporary audio file size: {file_size} bytes")
            
            if file_size < 1000:  # Less than 1KB is suspicious
                raise RuntimeError(f"Audio file too small: {file_size} bytes")
            
            # 🔧 CRITICAL FIX: Validate audio file before sending to Whisper
            try:
                # Quick validation by reading the file back
                test_audio, test_sr = sf.read(temp_path)
                if len(test_audio) < min_samples // 2:
                    raise RuntimeError(f"Written audio file too short: {len(test_audio)} samples")
                print(f"🔧 Audio file validation passed: {len(test_audio)} samples at {test_sr}Hz")
            except Exception as validation_error:
                print(f"⚠️ Audio file validation failed: {validation_error}")
                raise RuntimeError(f"Invalid audio file: {validation_error}")
            
            # Encode using audio2feat
            print("🔧 Calling audio_encoder.audio2feat()...")
            audio_features = audio_encoder.audio2feat(temp_path)
            
            if isinstance(audio_features, numpy.ndarray):
                audio_features = torch.from_numpy(audio_features)
            
            audio_features = audio_features.to(target_device).float()
            
            # 🔧 CRITICAL FIX: Handle Whisper's output shape properly
            # Whisper returns [seq_len, frames, embed_dim] but UNet expects [batch, seq_len, embed_dim]
            if audio_features.dim() == 3:
                # Reshape from [seq_len, frames, embed_dim] to [batch, seq_len*frames, embed_dim]
                seq_len, frames, embed_dim = audio_features.shape
                audio_features = audio_features.reshape(1, seq_len * frames, embed_dim)
                print(f"🔧 Reshaped Whisper output from [{seq_len}, {frames}, {embed_dim}] to {audio_features.shape}")
            elif audio_features.dim() == 2:
                audio_features = audio_features.unsqueeze(0)
            
            print(f"✅ Audio encoding successful: {audio_features.shape}")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return audio_features
            
        except Exception as encoding_error:
            print(f"⚠️ Audio encoding failed: {encoding_error}")
            print(f"🔧 Error type: {type(encoding_error)}")
            
            # 🔧 CRITICAL FIX: Check if this is the specific padding error
            if "Padding size should be less than" in str(encoding_error):
                print("🔧 Detected Whisper padding error - audio input too short for mel spectrogram")
                print("🔧 This usually happens with very short audio clips")
                
                # Try to create a longer audio file by repeating the input
                try:
                    print("🔧 Attempting to fix by creating longer audio...")
                    
                    # Create a much longer audio (5 seconds minimum for Whisper stability)
                    target_length = 16000 * 5  # 5 seconds
                    if len(audio_numpy) > 0:
                        repeat_count = (target_length // len(audio_numpy)) + 1
                        extended_audio = numpy.tile(audio_numpy, repeat_count)[:target_length]
                    else:
                        extended_audio = numpy.random.normal(0, 0.001, target_length).astype(numpy.float32)
                    
                    # Write the extended audio
                    sf.write(temp_path, extended_audio, 16000, format='WAV', subtype='PCM_16')
                    
                    print(f"🔧 Created extended audio: {len(extended_audio)} samples")
                    
                    # Try encoding again
                    audio_features = audio_encoder.audio2feat(temp_path)
                    
                    if isinstance(audio_features, numpy.ndarray):
                        audio_features = torch.from_numpy(audio_features)
                    
                    audio_features = audio_features.to(target_device).float()
                    
                    # 🔧 CRITICAL FIX: Handle Whisper's output shape properly
                    # Whisper returns [seq_len, frames, embed_dim] but UNet expects [batch, seq_len, embed_dim]
                    if audio_features.dim() == 3:
                        # Reshape from [seq_len, frames, embed_dim] to [batch, seq_len*frames, embed_dim]
                        seq_len, frames, embed_dim = audio_features.shape
                        audio_features = audio_features.reshape(1, seq_len * frames, embed_dim)
                        print(f"🔧 Reshaped Whisper output from [{seq_len}, {frames}, {embed_dim}] to {audio_features.shape}")
                    elif audio_features.dim() == 2:
                        audio_features = audio_features.unsqueeze(0)
                    
                    print(f"✅ Extended audio encoding successful: {audio_features.shape}")
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    return audio_features
                    
                except Exception as extended_error:
                    print(f"⚠️ Extended audio encoding also failed: {extended_error}")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as fallback_error:
        print(f"⚠️ Raw audio processing failed: {fallback_error}")
        print(f"🔧 Error type: {type(fallback_error)}")
    
    # 🔧 LAST RESORT: Create small dummy embeddings with warning
    print("⚠️ WARNING: Using dummy embeddings - this will affect lip sync quality")
    print("🔧 Consider using longer audio clips or pre-computed audio features")
    batch_size = 1
    seq_len = 10  # Small sequence length matching official slices
    embed_dim = 384  # Whisper Tiny embedding dimension
    
    dummy_embeddings = torch.zeros(batch_size, seq_len, embed_dim, device=target_device)
    print(f"✅ Created small dummy embeddings: {dummy_embeddings.shape}")
    
    return dummy_embeddings

def get_device_for_lip_syncer() -> str:
	"""Get the appropriate device for lip syncer operations"""
	return "cuda" if torch.cuda.is_available() else "cpu"

# 🎵 OFFICIAL LATENTSYNC AUDIO PROCESSING
def precompute_audio_features(audio_path: str) -> torch.Tensor:
    """
    Step 1: Pre-compute audio features once for entire file (Official LatentSync approach)
    This replaces the memory-intensive per-frame audio processing
    """
    global _audio_features_cache
    
    # Check cache first
    if audio_path in _audio_features_cache:
        print(f"✅ Using cached audio features for {audio_path}")
        return _audio_features_cache[audio_path]
    
    print(f"🎵 Pre-computing audio features for entire file: {audio_path}")
    
    try:
        audio_encoder = get_audio_encoder()
        
        # 🔧 OFFICIAL METHOD: Use audio2feat on entire file (this is efficient)
        whisper_features = audio_encoder.audio2feat(audio_path)
        
        print(f"✅ Pre-computed audio features shape: {whisper_features.shape}")
        print(f"   - This will be sliced into small chunks per frame")
        
        # Cache the features
        _audio_features_cache[audio_path] = whisper_features
        
        return whisper_features
        
    except Exception as e:
        print(f"❌ Failed to pre-compute audio features: {e}")
        # Return dummy features as fallback
        dummy_features = torch.zeros(100, 384)  # Small dummy features
        _audio_features_cache[audio_path] = dummy_features
        return dummy_features

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
    
    # 🔧 CRITICAL FIX: Better audio sync calculation
    # Use proper frame-to-audio mapping based on actual audio length
    audio_fps = 50.0  # Whisper's internal rate (50Hz)
    
    # Calculate center index with proper bounds checking
    center_idx = int(vid_idx * audio_fps / fps)
    
    # 🔧 CRITICAL FIX: Ensure center_idx is within audio bounds
    if center_idx >= length:
        # If beyond audio length, use the last valid audio frame
        center_idx = max(0, length - 1)
    
    # Calculate window bounds
    left_idx = center_idx - audio_feat_length[0] * 2
    right_idx = center_idx + (audio_feat_length[1] + 1) * 2
    
    # 🔧 OFFICIAL SLICING: Extract small window around current frame
    for idx in range(left_idx, right_idx):
        # 🔧 CRITICAL FIX: Better bounds handling
        if idx < 0:
            # Pad with first frame if before start
            actual_idx = 0
        elif idx >= length:
            # Pad with last frame if beyond end
            actual_idx = length - 1
        else:
            actual_idx = idx
        
        if actual_idx < len(feature_array):
            x = feature_array[actual_idx]
        else:
            # Fallback: create zeros if something goes wrong
            x = torch.zeros(embedding_dim, device=feature_array.device, dtype=feature_array.dtype)
        
        selected_feature.append(x)
    
    # 🔧 OFFICIAL FORMAT: Concatenate and reshape
    selected_feature = torch.cat(selected_feature, dim=0)
    selected_feature = selected_feature.reshape(-1, embedding_dim)  # Shape: [10, 384] - TINY!
    
    return selected_feature

def create_audio_chunks_official(audio_path: str, fps: float, total_frames: int) -> list:
    """
    Official LatentSync audio chunking workflow:
    1. Pre-compute features once
    2. Create small slices per frame
    """
    print(f"🎵 Creating audio chunks using official LatentSync method...")
    print(f"   - Total frames: {total_frames}")
    print(f"   - FPS: {fps}")
    
    # Step 1: Pre-compute features once (efficient)
    audio_features = precompute_audio_features(audio_path)
    
    # 🔧 CRITICAL ANALYSIS: Calculate audio coverage
    audio_fps = 50.0  # Whisper's internal rate (50Hz)
    audio_length_in_frames = len(audio_features)
    video_length_in_audio_frames = int(total_frames * audio_fps / fps)
    audio_duration_seconds = audio_length_in_frames / audio_fps
    video_duration_seconds = total_frames / fps
    
    print(f"🔧 Audio Analysis:")
    print(f"   - Audio features length: {audio_length_in_frames} frames")
    print(f"   - Audio duration: {audio_duration_seconds:.2f} seconds")
    print(f"   - Video duration: {video_duration_seconds:.2f} seconds")
    print(f"   - Video needs {video_length_in_audio_frames} audio frames")
    print(f"   - Audio coverage: {audio_length_in_frames >= video_length_in_audio_frames} ({audio_length_in_frames}/{video_length_in_audio_frames})")
    
    if audio_length_in_frames < video_length_in_audio_frames:
        shortage = video_length_in_audio_frames - audio_length_in_frames
        print(f"⚠️ Audio is {shortage} frames short ({shortage/audio_fps:.2f} seconds)")
        print(f"   - Frames {audio_length_in_frames}-{video_length_in_audio_frames-1} will use padding")
    
    # Step 2: Create small slices per frame (memory efficient)
    whisper_chunks = []
    for frame_idx in range(total_frames):
        audio_slice = get_sliced_audio_feature(audio_features, frame_idx, fps)
        whisper_chunks.append(audio_slice)
    
    print(f"✅ Created {len(whisper_chunks)} audio chunks")
    print(f"   - Each chunk shape: {whisper_chunks[0].shape if whisper_chunks else 'N/A'}")
    print(f"   - Memory per chunk: ~{whisper_chunks[0].numel() * 4 / 1024:.1f} KB (TINY!)")
    
    return whisper_chunks

def clear_audio_features_cache():
    """Clear the audio features cache to free memory"""
    global _audio_features_cache
    _audio_features_cache.clear()
    print("🧹 Cleared audio features cache")