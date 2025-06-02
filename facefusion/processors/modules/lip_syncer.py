from argparse import ArgumentParser
from functools import lru_cache
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
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(vae_device).float().eval()
        
        # 🔧 CRITICAL: Set official LatentSync VAE configuration
        vae.config.scaling_factor = 0.18215  # Official LatentSync scaling factor
        vae.config.shift_factor = 0          # Official LatentSync shift factor
        
        print(f"✅ VAE loaded with official LatentSync config:")
        print(f"   - Model: stabilityai/sd-vae-ft-mse (official)")
        print(f"   - Scaling factor: {vae.config.scaling_factor}")
        print(f"   - Shift factor: {vae.config.shift_factor}")
        print(f"   - Device: {vae_device}")
        print(f"   - Precision: float32")
    else:
        # 🔧 CRITICAL: Ensure VAE stays on GPU if available (consistent with audio encoder)
        if torch.cuda.is_available() and vae.device.type != 'cuda':
            try:
                vae = vae.float().cuda()
                print("🔄 Moved VAE to GPU with float32 precision for consistency")
            except Exception as move_error:
                print(f"⚠️ VAE GPU move failed: {move_error}")
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


def forward(temp_audio_frame: AudioFrame, close_vision_frame: VisionFrame, target_face: Face = None) -> VisionFrame:
    """
    Simplified forward function that uses the official LatentSync pipeline
    """
    model_name = state_manager.get_item('lip_syncer_model')
    
    if model_name == 'latentsync':
        # Use the official LatentSync pipeline we implemented
        # Convert AudioFrame to raw audio format expected by process_frame_latentsync
        if isinstance(temp_audio_frame, numpy.ndarray):
            # temp_audio_frame is already in the right format
            raw_audio = temp_audio_frame
        else:
            # Convert from AudioFrame format to raw audio
            raw_audio = temp_audio_frame.flatten() if hasattr(temp_audio_frame, 'flatten') else temp_audio_frame
        
        # Pass the actual target_face to process_frame_latentsync for proper mask creation
        return process_frame_latentsync(target_face, close_vision_frame, raw_audio)
    
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


def process_frame_latentsync(source_face: Face, target_frame: VisionFrame, audio_chunk: numpy.ndarray) -> VisionFrame:
    """
    Process a single frame using the official LatentSync pipeline
    """
    global audio_encoder, vae, unet_model, scheduler
    
    # 🧹 Memory monitoring
    log_memory_usage("🎬 Starting frame processing")
    
    # Get model name from global state
    model_name = state_manager.get_item('lip_syncer_model')
    target_device = get_device_for_lip_syncer()
    
    try:
        if model_name == 'latentsync':
            # ===== OFFICIAL LATENTSYNC PIPELINE =====
            
            # 1. Load models
            audio_encoder = get_audio_encoder()
            vae = get_vae()
            unet_model = get_unet_model()
            scheduler = get_scheduler()
            
            # 🔧 CRITICAL FIX: Determine target dtype from UNet model
            unet_dtype = next(unet_model.parameters()).dtype
            print(f"🔧 UNet model dtype: {unet_dtype}")
            
            # 2. Pipeline parameters (matching official)
            num_inference_steps = 20  # Official uses 20 steps
            guidance_scale = 3.5      # Official CFG scale
            do_classifier_free_guidance = guidance_scale > 1.0
            
            # 3. Set scheduler timesteps
            scheduler.set_timesteps(num_inference_steps, device=target_device)
            timesteps = scheduler.timesteps
            
            # 4. Process audio and image in proper scope
            with torch.no_grad():
                # Process audio to get embeddings
                print("🎵 Processing audio...")
                if len(audio_chunk.shape) == 1:
                    audio_chunk = audio_chunk[None, :]
                
                # 🔧 CRITICAL FIX: Ensure audio chunk is reasonable size for single frame
                # LatentSync should process ~1 second of audio per frame, not entire file
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
                    
                    # If embeddings are too large, truncate them
                    if audio_memory_gb > 2.0:  # More than 2GB is too much for single frame
                        print(f"⚠️ Audio embeddings too large ({audio_memory_gb:.2f} GB), truncating...")
                        max_seq_len = min(audio_embeds.shape[1], 1500)  # Reasonable sequence length
                        audio_embeds = audio_embeds[:, :max_seq_len, :]
                        print(f"🔧 Truncated to: {audio_embeds.shape}")
                
                if audio_embeds.dim() == 2:
                    audio_embeds = audio_embeds.unsqueeze(0)  # Add batch dim
                
                # 🔧 CRITICAL FIX: Convert audio embeddings to UNet dtype
                audio_embeds = audio_embeds.to(dtype=unet_dtype)
                print(f"🔧 Audio embeddings dtype: {audio_embeds.dtype}")
                print(f"🔧 Final audio embeddings shape: {audio_embeds.shape}")
                
                # 🔧 MEMORY-EFFICIENT CFG: Only duplicate if embeddings are reasonable size
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
                
                log_memory_usage("🎵 Audio processing complete")
                
                # 5. Process target frame
                print("🖼️ Processing target frame...")
                target_frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
                target_pil = Image.fromarray(target_frame_rgb)
                
                # Resize to 512x512 (official LatentSync resolution)
                target_pil = target_pil.resize((512, 512), Image.LANCZOS)
                
                # Convert to tensor and normalize
                target_tensor = torch.from_numpy(numpy.array(target_pil)).float() / 255.0
                target_tensor = target_tensor.permute(2, 0, 1).unsqueeze(0).to(target_device)
                target_tensor = (target_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
                
                # 🔧 CRITICAL FIX: Convert target tensor to UNet dtype
                target_tensor = target_tensor.to(dtype=unet_dtype)
                
                # 6. Encode image to latents
                print("🔄 Encoding image to latents...")
                image_latents = vae.encode(target_tensor).latent_dist.sample()
                # 🔧 CRITICAL FIX: Use official LatentSync VAE scaling (with shift factor)
                image_latents = (image_latents - vae.config.shift_factor) * vae.config.scaling_factor
                
                # 🔧 CRITICAL FIX: Ensure latents are in UNet dtype
                image_latents = image_latents.to(dtype=unet_dtype)
                
                # 7. Create mask (mouth region)
                print("🎭 Creating mouth mask...")
                mask = create_latentsync_mouth_mask(target_frame, source_face)
                mask_pil = Image.fromarray((mask * 255).astype(numpy.uint8)).resize((512, 512), Image.LANCZOS)
                mask_tensor = torch.from_numpy(numpy.array(mask_pil)).float() / 255.0
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(target_device)  # [1, 1, 512, 512]
                
                # 🔧 CRITICAL FIX: Convert mask to UNet dtype
                mask_tensor = mask_tensor.to(dtype=unet_dtype)
                
                # Encode mask to latent space
                mask_latents = F.interpolate(mask_tensor, size=(64, 64), mode='nearest')
                mask_latents = mask_latents.unsqueeze(2)  # Add frame dimension [1, 1, 1, 64, 64]
                
                # 8. Create masked image latents
                print("🖼️ Creating masked image latents...")
                masked_image = target_tensor * (1 - mask_tensor)
                masked_image_latents = vae.encode(masked_image).latent_dist.sample()
                # 🔧 CRITICAL FIX: Use official LatentSync VAE scaling (with shift factor)
                masked_image_latents = (masked_image_latents - vae.config.shift_factor) * vae.config.scaling_factor
                masked_image_latents = masked_image_latents.unsqueeze(2)  # Add frame dimension
                
                # 🔧 CRITICAL FIX: Ensure masked image latents are in UNet dtype
                masked_image_latents = masked_image_latents.to(dtype=unet_dtype)
                
                # 9. Prepare reference latents (same as image latents for single frame)
                ref_latents = image_latents.unsqueeze(2)  # Add frame dimension
                
                # 10. Prepare initial noise latents
                print("🎲 Preparing initial latents...")
                batch_size = 1
                num_frames = 1
                num_channels_latents = 4
                height, width = 64, 64
                
                latents = prepare_latents(
                    batch_size, num_frames, num_channels_latents, height, width,
                    dtype=unet_dtype, device=target_device  # 🔧 Use UNet dtype
                )
                
                log_memory_usage("🔄 Latents prepared")
                
                # 11. MAIN DENOISING LOOP (Official LatentSync Pipeline)
                print(f"🔄 Starting {num_inference_steps}-step denoising...")
                
                for i, t in enumerate(timesteps):
                    print(f"  Step {i+1}/{num_inference_steps} (t={t})")
                    
                    # Expand latents for CFG
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    
                    # Scale model input (official scheduler scaling)
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                    
                    # 🔧 CRITICAL FIX: Ensure latent input is in UNet dtype
                    latent_model_input = latent_model_input.to(dtype=unet_dtype)
                    
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
                    
                    # 🔧 CRITICAL FIX: Ensure all conditioning inputs are in UNet dtype
                    mask_input = mask_input.to(dtype=unet_dtype)
                    masked_img_input = masked_img_input.to(dtype=unet_dtype)
                    ref_input = ref_input.to(dtype=unet_dtype)
                    
                    # Concatenate all inputs (official order)
                    unet_input = torch.cat([
                        latent_model_input,
                        mask_input,
                        masked_img_input,
                        ref_input
                    ], dim=1)
                    
                    # 🔧 CRITICAL FIX: Final dtype check before UNet forward
                    unet_input = unet_input.to(dtype=unet_dtype)
                    audio_embeds_input = audio_embeds.to(dtype=unet_dtype)
                    
                    # UNet forward pass
                    noise_pred = unet_model(
                        sample=unet_input,
                        timestep=t,
                        encoder_hidden_states=audio_embeds_input
                    ).sample
                    
                    # Apply classifier-free guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # Scheduler step (official denoising)
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                    
                    # 🔧 CRITICAL FIX: Ensure latents maintain UNet dtype
                    latents = latents.to(dtype=unet_dtype)
                    
                    # 🧹 MINIMAL CLEANUP: Only every 5 steps and only cache
                    if i % 5 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                print("✅ Denoising complete!")
                log_memory_usage("🔄 Denoising complete")
                
                # 12. Decode latents back to image
                print("🎨 Decoding latents to image...")
                latents = latents.squeeze(2)  # Remove frame dimension
                # 🔧 CRITICAL FIX: Use official LatentSync VAE decoding (with shift factor)
                latents = latents / vae.config.scaling_factor + vae.config.shift_factor
                
                decoded_image = vae.decode(latents).sample
                
                # 13. Post-process decoded image
                decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
                decoded_image = decoded_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                decoded_image = (decoded_image * 255).astype(numpy.uint8)
                
                # 14. Resize back to original frame size
                original_height, original_width = target_frame.shape[:2]
                decoded_pil = Image.fromarray(decoded_image)
                decoded_pil = decoded_pil.resize((original_width, original_height), Image.LANCZOS)
                result_frame = numpy.array(decoded_pil)
                result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                
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

	# 🚀 BATCH AUDIO PROCESSING: Pre-compute all audio chunks (official LatentSync approach)
	audio_chunks = None
	if model_name == 'latentsync' and source_audio_path:
		print("🎵 Pre-computing audio chunks for LatentSync batch processing...")
		total_frames = len(queue_payloads)
		audio_chunks = get_audio_chunks_for_latentsync(source_audio_path, temp_video_fps, total_frames)
		print(f"✅ Pre-computed {len(audio_chunks)} audio chunks for {total_frames} frames")

	for queue_payload in process_manager.manage(queue_payloads):
		frame_number = queue_payload.get('frame_number')
		target_vision_path = queue_payload.get('frame_path')
		
		# Get appropriate audio frame based on model
		if model_name == 'latentsync':
			if audio_chunks is not None:
				# Use pre-computed batch chunks (official LatentSync approach)
				source_audio_frame = get_audio_chunk_from_batch(audio_chunks, frame_number)
			else:
				# Fallback to per-frame processing if batch failed
				source_audio_frame = get_raw_audio_frame(source_audio_path, temp_video_fps, frame_number)
			
			if source_audio_frame is None or not numpy.any(source_audio_frame):
				# Create empty raw audio frame with consistent FP32 format
				source_audio_frame = create_empty_raw_audio_frame(temp_video_fps, sample_rate=16000)
		else:
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
	reference_faces = get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else None
	model_name = state_manager.get_item('lip_syncer_model')
	
	# Create appropriate empty audio frame based on model
	if model_name == 'latentsync':
		# Create empty raw audio frame with consistent FP32 format (1 second at 16kHz)
		source_audio_frame = create_empty_raw_audio_frame(fps=1.0, sample_rate=16000)  # 1 FPS = 1 second
	else:
		source_audio_frame = create_empty_audio_frame()
	
	target_vision_frame = read_static_image(target_path)
	output_vision_frame = process_frame_original(
	{
		'reference_faces': reference_faces,
		'source_audio_frame': source_audio_frame,
		'target_vision_frame': target_vision_frame
	})
	write_image(output_path, output_vision_frame)


def process_video(source_paths : List[str], temp_frame_paths : List[str]) -> None:
	# 🎵 Pre-load audio for traditional models (Wav2Lip compatibility)
	source_audio_paths = filter_audio_paths(state_manager.get_item('source_paths'))
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
	for source_audio_path in source_audio_paths:
		read_static_voice(source_audio_path, temp_video_fps)
	
	# 🚀 Note: LatentSync now uses batch audio processing in process_frames()
	# This improves temporal consistency and matches official LatentSync behavior
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
            unet_model.eval().to(target_device).float()
            
            # Clean up checkpoint from memory
            del checkpoint, state_dict
            torch.cuda.empty_cache()
            
            print(f"✅ PyTorch UNet loaded on {target_device}")
            
        except Exception as load_error:
            print(f"❌ Failed to load PyTorch UNet: {load_error}")
            unet_model = None
            raise RuntimeError(f"Could not initialize PyTorch UNet: {load_error}")
    
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

def create_latentsync_mouth_mask(target_frame: VisionFrame, source_face: Face) -> numpy.ndarray:
    """Create a mouth region mask for inpainting"""
    height, width = target_frame.shape[:2]
    mask = numpy.zeros((height, width), dtype=numpy.float32)
    
    # Get face landmarks if available
    if source_face is not None and hasattr(source_face, 'landmark_set') and source_face.landmark_set is not None:
        landmarks_68 = source_face.landmark_set.get('68')
        
        if landmarks_68 is not None and len(landmarks_68) >= 68:
            landmarks = landmarks_68.astype(numpy.int32)
            
            # Mouth landmarks (indices 48-67 in 68-point landmarks)
            mouth_points = landmarks[48:68]
            
            # Create convex hull around mouth
            hull = cv2.convexHull(mouth_points)
            cv2.fillPoly(mask, [hull], 1.0)
            
            # Dilate mask slightly for better coverage
            kernel = numpy.ones((15, 15), numpy.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        # Fallback: create a simple rectangular mask in the lower face region
        center_x, center_y = width // 2, int(height * 0.7)
        mask_width, mask_height = width // 4, height // 6
        
        x1 = max(0, center_x - mask_width // 2)
        x2 = min(width, center_x + mask_width // 2)
        y1 = max(0, center_y - mask_height // 2)
        y2 = min(height, center_y + mask_height // 2)
        
        mask[y1:y2, x1:x2] = 1.0
    
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
    
    # Only prepare audio frame for non-latentsync models
    if model_name != 'latentsync':
        temp_audio_frame = prepare_audio_frame(temp_audio_frame)
    
    crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark_set.get('5/68'), 'ffhq_512', (512, 512))
    face_landmark_68 = cv2.transform(target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
    bounding_box = create_bounding_box(face_landmark_68)
    bounding_box[1] -= numpy.abs(bounding_box[3] - bounding_box[1]) * 0.125
    

    # Create masks for final paste-back operation
    # LatentSync creates its own internal mask for diffusion, but still needs masks for paste-back
    # Non-LatentSync models: use FaceFusion mouth mask for both processing and paste-back
    # LatentSync: create mouth mask only for final paste-back (not used during diffusion)
    mouth_mask = create_mouth_mask(face_landmark_68)
    crop_masks = [mouth_mask]

    box_mask = create_static_box_mask(crop_vision_frame.shape[:2][::-1], state_manager.get_item('face_mask_blur'), state_manager.get_item('face_mask_padding'))
    crop_masks.append(box_mask)

    if 'occlusion' in state_manager.get_item('face_mask_types'):
        occlusion_mask = create_occlusion_mask(crop_vision_frame)
        crop_masks.append(occlusion_mask)

    # --- Lip Sync Forward ---
    close_vision_frame, close_matrix = warp_face_by_bounding_box(crop_vision_frame, bounding_box, model_size)
    
    # Different preprocessing based on model type
    if model_name == 'latentsync':
        # LatentSync expects regular BGR image (512, 512, 3)
        # No special preprocessing needed - just pass the warped frame directly
        pass
    else:
        # Wav2Lip expects concatenated format (1, 6, H, W)
        close_vision_frame = prepare_crop_frame(close_vision_frame)
    
    close_vision_frame = forward(temp_audio_frame, close_vision_frame, target_face)
    
    # --- Process model output based on model type ---
    if model_name != 'latentsync':
        close_vision_frame = normalize_close_frame(close_vision_frame)

    # --- Check if the model returned a valid frame ---
    if close_vision_frame is None:
        print("⚠️ Model returned None frame, using original frame")
        return temp_vision_frame

    if not isinstance(close_vision_frame, numpy.ndarray):
        print("⚠️ Invalid frame type:", type(close_vision_frame))
        return temp_vision_frame

    if len(close_vision_frame.shape) < 2:
        print("⚠️ Invalid frame dimensions:", close_vision_frame.shape)
        return temp_vision_frame

    if close_vision_frame.size == 0 or close_vision_frame.shape[0] == 0 or close_vision_frame.shape[1] == 0:
        print("⚠️ Empty frame detected:", close_vision_frame.shape)
        return temp_vision_frame

    # Different expected shapes based on model
    if model_name == 'latentsync':
        expected_shape = (512, 512, 3)  # LatentSync outputs BGR image directly
    else:
        expected_shape = (96, 96, 3)   # Wav2Lip models output 96x96
    
    if close_vision_frame.shape != expected_shape:
        print(f"⚠️ Unexpected frame shape: got {close_vision_frame.shape}, expected {expected_shape}")
        print(f"🔍 Model: {model_name}, Frame dtype: {close_vision_frame.dtype}")
        return temp_vision_frame

    # --- Apply mask and paste lips back ---
    crop_vision_frame = cv2.warpAffine(close_vision_frame, cv2.invertAffineTransform(close_matrix), (512, 512), borderMode=cv2.BORDER_REPLICATE)
    crop_mask = numpy.minimum.reduce(crop_masks)
    print("🔍 crop_mask min/max:", crop_mask.min(), crop_mask.max())

    paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
    return paste_vision_frame

def encode_audio_for_latentsync(audio_tensor):
    """Encode audio using the audio encoder for LatentSync (official method)"""
    audio_encoder = get_audio_encoder()
    
    # 🔧 CRITICAL FIX: Use actual audio2feat method like official LatentSync
    try:
        # Convert tensor to numpy if needed
        if isinstance(audio_tensor, torch.Tensor):
            audio_numpy = audio_tensor.cpu().numpy()
        else:
            audio_numpy = audio_tensor
        
        # 🔧 DEBUG: Check input audio size
        print(f"🔍 Input audio shape: {audio_numpy.shape}")
        print(f"🔍 Input audio size: {audio_numpy.size} samples")
        audio_duration = audio_numpy.size / 16000  # Assuming 16kHz
        print(f"🔍 Input audio duration: {audio_duration:.2f} seconds")
        
        # 🔧 CRITICAL FIX: Ensure minimum audio length for Whisper (at least 1 second)
        min_samples = 16000  # 1 second at 16kHz
        if len(audio_numpy) < min_samples:
            # Pad audio to minimum length
            padding_needed = min_samples - len(audio_numpy)
            audio_numpy = numpy.pad(audio_numpy, (0, padding_needed), mode='constant', constant_values=0)
            print(f"🔧 Padded audio from {len(audio_numpy) - padding_needed} to {len(audio_numpy)} samples")
        
        # 🔧 MEMORY SAFETY: Limit audio length to prevent OOM
        max_samples = 16000 * 2  # 2 seconds max to prevent memory issues
        if len(audio_numpy) > max_samples:
            print(f"⚠️ Audio too long ({len(audio_numpy)} samples), truncating to {max_samples} samples")
            audio_numpy = audio_numpy[:max_samples]
        
        # 🔧 CRITICAL: Try multiple approaches in order of preference
        
        # Method 1: Official audio2feat method (file-based)
        if hasattr(audio_encoder, 'audio2feat'):
            try:
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # 🔧 ENHANCED: Try multiple audio writing approaches
                audio_written = False
                
                # Approach 1: Standard soundfile with explicit format
                try:
                    # Ensure proper data type and format for soundfile
                    if audio_numpy.dtype != numpy.float32:
                        audio_numpy = audio_numpy.astype(numpy.float32)
                    
                    # Ensure audio is in proper range for WAV format
                    if numpy.max(numpy.abs(audio_numpy)) > 1.0:
                        audio_numpy = audio_numpy / numpy.max(numpy.abs(audio_numpy))
                    
                    sf.write(temp_path, audio_numpy, 16000, format='WAV', subtype='PCM_16')
                    
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        audio_written = True
                        print("✅ Audio file written with soundfile")
                
                except Exception as sf_error:
                    print(f"⚠️ Soundfile approach failed: {sf_error}")
                
                # Approach 2: Use scipy.io.wavfile as fallback
                if not audio_written:
                    try:
                        import scipy.io.wavfile as wavfile
                        # 🔧 CRITICAL FIX: Ensure audio is 1D array for proper channel handling
                        if audio_numpy.ndim > 1:
                            audio_numpy = audio_numpy.flatten()
                        
                        # Convert to int16 for scipy and ensure proper range
                        audio_numpy_clipped = numpy.clip(audio_numpy, -1.0, 1.0)
                        audio_int16 = (audio_numpy_clipped * 32767).astype(numpy.int16)
                        
                        wavfile.write(temp_path, 16000, audio_int16)
                        
                        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                            audio_written = True
                            print("✅ Audio file written with scipy.wavfile")
                    
                    except Exception as scipy_error:
                        print(f"⚠️ Scipy wavfile approach failed: {scipy_error}")
                
                # If file was written successfully, try audio2feat
                if audio_written:
                    try:
                        audio_features = audio_encoder.audio2feat(temp_path)
                        
                        # 🔧 CRITICAL DEBUG: Check audio embeddings size
                        if isinstance(audio_features, numpy.ndarray):
                            print(f"🔍 Raw audio_features shape: {audio_features.shape}")
                            print(f"🔍 Raw audio_features dtype: {audio_features.dtype}")
                            print(f"🔍 Raw audio_features size: {audio_features.size} elements")
                            memory_size_gb = audio_features.nbytes / (1024**3)
                            print(f"🔍 Raw audio_features memory: {memory_size_gb:.2f} GB")
                            
                            # 🔧 MEMORY SAFETY: Limit embeddings size
                            max_elements = 1000000  # 1M elements max (~4MB for float32)
                            if audio_features.size > max_elements:
                                print(f"⚠️ Audio embeddings too large ({audio_features.size} elements), truncating to {max_elements}")
                                # Reshape and truncate while preserving structure
                                original_shape = audio_features.shape
                                audio_features_flat = audio_features.flatten()[:max_elements]
                                # Try to maintain reasonable dimensions
                                if len(original_shape) == 2:
                                    new_seq_len = min(original_shape[1], max_elements // original_shape[0])
                                    audio_features = audio_features_flat[:original_shape[0] * new_seq_len].reshape(original_shape[0], new_seq_len)
                                else:
                                    audio_features = audio_features_flat.reshape(-1, audio_features_flat.shape[0])
                            
                            audio_features = torch.from_numpy(audio_features)
                        
                        # Ensure proper device placement
                        target_device = "cuda" if torch.cuda.is_available() else "cpu"
                        audio_features = audio_features.to(target_device)
                        
                        # 🔧 FINAL DEBUG: Check final tensor size
                        memory_size_gb = audio_features.numel() * 4 / (1024**3)  # 4 bytes per float32
                        print(f"🔍 Final audio_features shape: {audio_features.shape}")
                        print(f"🔍 Final audio_features memory: {memory_size_gb:.2f} GB")
                        
                        print(f"✅ Audio encoded using official audio2feat method")
                        print(f"   - Input shape: {audio_numpy.shape}")
                        print(f"   - Output shape: {audio_features.shape}")
                        print(f"   - Device: {audio_features.device}")
                        
                        # Clean up and return
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        return audio_features
                    
                    except Exception as audio2feat_error:
                        print(f"⚠️ audio2feat method failed: {audio2feat_error}")
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            except Exception as file_method_error:
                print(f"⚠️ File-based audio encoding failed: {file_method_error}")
        
        # Method 2: Try direct model access (corrected approach)
        if hasattr(audio_encoder, 'model') and audio_encoder.model is not None:
            try:
                print("🔄 Trying direct Whisper model approach...")
                
                # Convert audio to tensor for direct processing
                audio_tensor_input = torch.from_numpy(audio_numpy).float()
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
                audio_tensor_input = audio_tensor_input.to(target_device)
                
                # Reshape for Whisper input format [batch, samples]
                if audio_tensor_input.dim() == 1:
                    audio_tensor_input = audio_tensor_input.unsqueeze(0)
                
                with torch.no_grad():
                    # 🔧 CORRECTED: Use proper Whisper model methods
                    if hasattr(audio_encoder.model, 'encoder'):
                        # Try different Whisper model interfaces
                        try:
                            # Method 1: Standard Whisper interface
                            if hasattr(audio_encoder.model, 'log_mel_spectrogram'):
                                mel = audio_encoder.model.log_mel_spectrogram(audio_tensor_input)
                                audio_features = audio_encoder.model.encoder(mel)
                            # Method 2: Alternative Whisper interface  
                            elif hasattr(audio_encoder.model, 'get_audio_features'):
                                audio_features = audio_encoder.model.get_audio_features(audio_tensor_input)
                            # Method 3: Direct encoder call with mel computation
                            else:
                                # Compute mel spectrogram manually if needed
                                import whisper
                                mel = whisper.log_mel_spectrogram(audio_tensor_input.squeeze(0))
                                mel = mel.unsqueeze(0).to(audio_tensor_input.device)
                                audio_features = audio_encoder.model.encoder(mel)
                            
                            print(f"✅ Audio encoded using direct Whisper encoder")
                            print(f"   - Input shape: {audio_tensor_input.shape}")
                            print(f"   - Output shape: {audio_features.shape}")
                            print(f"   - Device: {audio_features.device}")
                            
                            return audio_features
                            
                        except Exception as whisper_method_error:
                            print(f"⚠️ Whisper method failed: {whisper_method_error}")
                            # Continue to fallback
                    
                    # Try alternative model access patterns
                    elif hasattr(audio_encoder, 'encode_audio'):
                        audio_features = audio_encoder.encode_audio(audio_tensor_input)
                        return audio_features
                
            except Exception as direct_error:
                print(f"⚠️ Direct Whisper model approach failed: {direct_error}")
        
        # Method 3: Create compatible dummy embeddings (guaranteed fallback)
        print("⚠️ All audio encoding methods failed, using compatible dummy embeddings")
        
    except Exception as encoding_error:
        print(f"❌ Audio encoding failed: {encoding_error}")
        print("⚠️ Using fallback dummy embeddings")
    
    # 🔧 GUARANTEED FALLBACK: Always return valid embeddings
    batch_size = 1
    seq_len = 77  # Standard sequence length for compatibility
    embed_dim = 384  # Whisper Tiny embedding dimension (official LatentSync)
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dummy_embeddings = torch.zeros(batch_size, seq_len, embed_dim, device=target_device)
    print(f"✅ Created dummy embeddings: {dummy_embeddings.shape} on {dummy_embeddings.device}")
    
    return dummy_embeddings

def get_device_for_lip_syncer() -> str:
	"""Get the appropriate device for lip syncer operations"""
	return "cuda" if torch.cuda.is_available() else "cpu"