from argparse import ArgumentParser
from functools import lru_cache
from typing import List

import cv2
import numpy
import torch
import torch.nn.functional as F
import librosa
import scipy.signal
import soundfile as sf

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
import facefusion.processors.core as processors
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, inference_manager, logger, process_manager, state_manager, voice_extractor, wording
from facefusion.audio import create_empty_audio_frame, get_voice_frame, read_static_voice, get_raw_audio_frame, create_empty_raw_audio_frame
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
    LATENTSYNC_AVAILABLE = True
except ImportError:
    print("⚠️ LatentSync not available. LatentSync model will be disabled.")
    Audio2Feature = None
    LATENTSYNC_AVAILABLE = False

# 🧹 MEMORY OPTIMIZATION: Lazy model loading to prevent OOM
# Only load models when needed, not at import time
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global model variables (loaded lazily)
audio_encoder = None
vae = None
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
            
        # 🔧 Use float32 for better GPU compatibility
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(vae_device).float().eval()
        print(f"✅ VAE loaded with float32 precision on {vae_device}.")
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
					'url': resolve_download_url('models-3.0.0', 'latentsync_unet.hash'),
					'path': resolve_relative_path('../.assets/models/latentsync_model_files/latentsync_unet.hash')
				}
			},
			'sources':
			{
				'lip_syncer':
				{
					'url': resolve_download_url('models-3.0.0', 'latentsync_unet.onnx'),
					'path': resolve_relative_path('../.assets/models/latentsync_model_files/latentsync_unet.onnx')
				}
			},
			'size': (512, 512)
		}
	}


def get_inference_pool() -> InferencePool:
	model_sources = get_model_options().get('sources')
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
    mouth_mask = create_mouth_mask(face_landmark_68)
    box_mask = create_static_box_mask(crop_vision_frame.shape[:2][::-1], state_manager.get_item('face_mask_blur'), state_manager.get_item('face_mask_padding'))
    crop_masks = [
        mouth_mask,
        box_mask
    ]

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
    
    close_vision_frame = forward(temp_audio_frame, close_vision_frame)
    
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


def forward(temp_audio_frame: AudioFrame, close_vision_frame: VisionFrame) -> VisionFrame:
    global audio_encoder, vae  # Move global declarations to top
    
    lip_syncer = None  # Initialize to None
    model_name = state_manager.get_item('lip_syncer_model')
    
    print(f"🔍 Forward function called with model: {model_name}")
    print(f"🔍 Input close_vision_frame shape: {close_vision_frame.shape}")
    print(f"🔍 Input close_vision_frame dtype: {close_vision_frame.dtype}")

    # 🧹 AGGRESSIVE GPU MEMORY CLEANUP before loading ONNX model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 🔥 COMPLETE DEVICE RESET to prevent mixed device states
        print("🔄 Performing complete device reset to prevent mixed states...")
        reset_models_to_device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check available memory before ONNX loading
        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        available_gb = available_memory / 1024**3
        print(f"💾 Available GPU memory before ONNX load: {available_gb:.1f} GB")
        
        if available_gb < 3.0:  # Need at least 3GB for LatentSync ONNX model
            print("⚠️ Low GPU memory detected! Forcing aggressive cleanup...")
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Try to free more memory
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            available_gb = available_memory / 1024**3
            print(f"💾 Available GPU memory after cleanup: {available_gb:.1f} GB")
            
            if available_gb < 2.0:
                raise RuntimeError(f"Insufficient GPU memory: {available_gb:.1f} GB available, need at least 2GB for LatentSync")
        
        # Emergency reset if we detect any mixed device issues
        try:
            # Test device consistency
            if audio_encoder is not None and hasattr(audio_encoder, 'model'):
                model_devices = set()
                for param in audio_encoder.model.parameters():
                    model_devices.add(param.device.type)
                if len(model_devices) > 1:
                    print(f"⚠️ Mixed devices detected in audio encoder: {model_devices}")
                    reset_models_to_device("cpu")
        except Exception as device_check_error:
            print(f"⚠️ Device check error: {device_check_error}")
            reset_models_to_device("cpu")

    try:
        lip_syncer = get_inference_pool().get('lip_syncer')  # ONNX Runtime session
    except Exception as onnx_error:
        print(f"❌ Failed to load ONNX model: {onnx_error}")
        if "Failed to allocate memory" in str(onnx_error):
            print("💡 Suggestion: Clear GPU memory and restart the application")
            print("💡 Run: sudo kill <other_python_processes> or restart")
        raise

    with conditional_thread_semaphore():
        if model_name == 'latentsync':
            if not LATENTSYNC_AVAILABLE:
                logger.error("LatentSync model selected but LatentSync is not available!", __name__)
                return close_vision_frame
                
            print("🚀 Executing LatentSync path...")
            try:
                with torch.no_grad():
                    # 🧹 AGGRESSIVE MEMORY CLEANUP at start
                    torch.cuda.empty_cache()
                    
                    # 🔧 CRITICAL: Determine target device FIRST and ensure ALL models use it
                    target_device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"🔧 Target device determined: {target_device}")
                    
                    # 🔧 ENSURE ALL MODELS ARE ON THE SAME DEVICE
                    print("🔧 Ensuring all models are on the same device...")
                    
                    # Move audio encoder to target device if needed
                    if audio_encoder is not None and hasattr(audio_encoder, 'model') and audio_encoder.model is not None:
                        current_audio_device = audio_encoder.model.device
                        if current_audio_device.type != target_device:
                            print(f"🔧 Moving audio encoder from {current_audio_device} to {target_device}")
                            audio_encoder.model = audio_encoder.model.to(target_device).float()
                            # Ensure ALL parameters are on target device
                            for name, param in audio_encoder.model.named_parameters():
                                if param.device.type != target_device:
                                    param.data = param.data.to(target_device)
                            # Ensure ALL buffers are on target device  
                            for name, buffer in audio_encoder.model.named_buffers():
                                if buffer.device.type != target_device:
                                    buffer.data = buffer.data.to(target_device)
                    
                    # Move VAE to target device if needed
                    vae_instance = get_vae()
                    current_vae_device = vae_instance.device
                    if current_vae_device.type != target_device:
                        print(f"🔧 Moving VAE from {current_vae_device} to {target_device}")
                        if target_device == "cpu":
                            vae_instance = vae_instance.cpu().float()
                        else:
                            vae_instance = vae_instance.to(target_device).float()
                        # Update global VAE reference
                        global vae
                        vae = vae_instance
                    
                    print(f"✅ All models confirmed on {target_device}")
                    
                    # 🔧 CRITICAL: Verify device consistency between all models
                    try:
                        common_device = verify_device_consistency()
                        if common_device != target_device:
                            print(f"⚠️ Warning: Models on {common_device} but target is {target_device}")
                    except Exception as consistency_error:
                        print(f"❌ Device consistency check failed: {consistency_error}")
                        # Try to fix by forcing all models to target device
                        reset_models_to_device(target_device)
                    
                    # 🧹 AGGRESSIVE MEMORY CLEANUP at start
                    torch.cuda.empty_cache()
                    
                    # 🔧 ENSURE COMPLETE DEVICE CONSISTENCY
                    print("🔧 Ensuring complete device consistency for audio processing...")
                    
                    # Reset everything to target device
                    reset_models_to_device(target_device)
                    
                    # 💾 Check available memory and use ULTRA-CONSERVATIVE settings
                    if torch.cuda.is_available() and target_device == "cuda":
                        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                        available_gb = available_memory / 1024**3
                        print(f"💾 Available GPU memory: {available_gb:.1f} GB")
                        
                        # 🔥 ULTRA-CONSERVATIVE SETTINGS for memory constrained system
                        if available_gb < 15.0:  # Less than 15GB available (very conservative)
                            print("⚠️ Memory constrained! Using ultra-minimal settings.")
                            guidance_scale = 1.0  # Disable CFG completely
                            num_inference_steps = 3   # Absolute minimum steps
                        else:
                            print("🚀 Good memory detected! Using standard settings.")
                            guidance_scale = 1.2  # Reduced CFG
                            num_inference_steps = 8  # Reduced steps
                    else:
                        guidance_scale = 1.0
                        num_inference_steps = 3
                    
                    do_classifier_free_guidance = guidance_scale > 1.0
                    
                    # 🏗️ STEP 1: Prepare audio with immediate cleanup
                    print(f"🔍 Input temp_audio_frame shape: {temp_audio_frame.shape if hasattr(temp_audio_frame, 'shape') else 'No shape attr'}")
                    print(f"🔍 Input temp_audio_frame type: {type(temp_audio_frame)}")
                    audio_tensor = prepare_latentsync_audio(temp_audio_frame, sample_rate=16000)
                    print(f"🔍 Audio tensor shape: {audio_tensor.shape}")
                    print(f"🔍 Audio tensor device: {audio_tensor.device}")
                    
                    # 🔧 CRITICAL: Ensure audio tensor is on target device
                    if audio_tensor.device.type != target_device:
                        print(f"🔧 Moving audio tensor from {audio_tensor.device} to {target_device}")
                        audio_tensor = audio_tensor.to(target_device)
                    
                    # 🧹 IMMEDIATELY offload audio encoder to CPU to save memory
                    if audio_encoder is not None and hasattr(audio_encoder, 'model'):
                        # 🔧 Keep audio encoder on target device for device consistency
                        torch.cuda.empty_cache()
                        
                        # Apply classifier-free guidance for audio only if needed
                        if do_classifier_free_guidance:
                            null_audio_embeds = torch.zeros_like(audio_tensor)
                            audio_tensor = torch.cat([null_audio_embeds, audio_tensor])
                            print(f"🔍 CFG Audio tensor shape: {audio_tensor.shape}")
                            del null_audio_embeds
                            torch.cuda.empty_cache()
                                         
                    # 🏗️ STEP 2: Prepare video with VAE cleanup
                    video_latent = prepare_latentsync_frame(close_vision_frame)
                    print(f"🔍 Video latent device: {video_latent.device}")
                    
                    # 🔧 CRITICAL: Ensure video latent is on target device
                    if video_latent.device.type != target_device:
                        print(f"🔧 Moving video latent from {video_latent.device} to {target_device}")
                        video_latent = video_latent.to(target_device)
                    
                    # 🧹 Clear cache after VAE encoding (keep VAE on target device for consistency)
                    torch.cuda.empty_cache()
                    
                    # 🏗️ STEP 3: Setup minimal diffusion (ultra-conservative)
                    noise = torch.randn(1, 4, 1, 64, 64, dtype=torch.float32, device=target_device)
                    
                    # Simple mouth mask (no complex masking to save memory)
                    mask_height, mask_width = 64, 64
                    mouth_mask = torch.zeros((mask_height, mask_width), dtype=torch.float32, device=target_device)
                    mouth_y_start, mouth_y_end = int(mask_height * 0.6), int(mask_height * 0.9)
                    mouth_x_start, mouth_x_end = int(mask_width * 0.3), int(mask_width * 0.7)
                    mouth_mask[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end] = 1.0
                    
                    # 🔧 CRITICAL FIX: Ensure mask has correct 5D shape (1, 1, 1, 64, 64)
                    mask_latents = mouth_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, 64, 64)
                    print(f"🔍 Initial mask_latents shape: {mask_latents.shape}")
                    
                    if do_classifier_free_guidance:
                        mask_latents = torch.cat([mask_latents] * 2)  # (2, 1, 1, 64, 64)
                        print(f"🔍 CFG mask_latents shape: {mask_latents.shape}")
                    
                    # Create masked image latents
                    video_latent_5d = video_latent.unsqueeze(2)  # (1, 4, 1, 64, 64)
                    print(f"🔍 video_latent_5d shape: {video_latent_5d.shape}")
                    
                    # 🔧 CRITICAL FIX: Expand mask to match video latent channels (4 channels)
                    mask_expanded = mask_latents.repeat(1, 4, 1, 1, 1)  # (1, 4, 1, 64, 64) or (2, 4, 1, 64, 64)
                    print(f"🔍 mask_expanded shape: {mask_expanded.shape}")
                    
                    masked_image_latents = video_latent_5d * (1 - mask_expanded)
                    print(f"🔍 masked_image_latents shape: {masked_image_latents.shape}")
                    
                    if do_classifier_free_guidance:
                        masked_image_latents = torch.cat([masked_image_latents] * 2)
                        print(f"🔍 CFG masked_image_latents shape: {masked_image_latents.shape}")
                    
                    ref_latents = video_latent_5d
                    if do_classifier_free_guidance:
                        ref_latents = torch.cat([ref_latents] * 2)
                    print(f"🔍 ref_latents shape: {ref_latents.shape}")
                    
                    # Clean up intermediate tensors immediately
                    del video_latent, video_latent_5d, mask_expanded, mouth_mask
                    torch.cuda.empty_cache()
                    
                    # 🏗️ STEP 4: Single denoising step (minimal)
                    timestep = torch.tensor([50], dtype=torch.long, device=target_device)  # Single timestep
                    
                    # Use noise as initial latents
                    latents = noise
                    print(f"🔍 Initial latents shape: {latents.shape}")
                    
                    # Prepare inputs for UNet
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    print(f"🔍 latent_model_input shape: {latent_model_input.shape}")
                    
                    # 🔧 CRITICAL FIX: Ensure mask_input has correct shape for concatenation
                    mask_input = mask_latents  # Use mask_latents directly (already has correct CFG handling)
                    print(f"🔍 mask_input shape: {mask_input.shape}")
                    
                    # Move all inputs to same device
                    latent_model_input = latent_model_input.to(target_device)
                    mask_input = mask_input.to(target_device)
                    audio_tensor = audio_tensor.to(target_device)
                    
                    # 🔧 CRITICAL: Verify all tensor shapes before concatenation
                    print(f"🔍 Pre-concat shapes:")
                    print(f"   latent_model_input: {latent_model_input.shape}")
                    print(f"   mask_input: {mask_input.shape}")
                    print(f"   masked_image_latents: {masked_image_latents.shape}")
                    print(f"   ref_latents: {ref_latents.shape}")
                    
                    # 🏗️ STEP 5: Single denoising step with aggressive cleanup
                    torch.cuda.empty_cache()  # Clean before step
                    
                    # 🔧 CRITICAL FIX: Prepare concatenated inputs with proper channel handling
                    # Expected UNet input: (batch, 13, frames, height, width)
                    # latent_model_input: (batch, 4, 1, 64, 64)
                    # mask_input: (batch, 1, 1, 64, 64) 
                    # masked_image_latents: (batch, 4, 1, 64, 64)
                    # ref_latents: (batch, 4, 1, 64, 64)
                    # Total: 4 + 1 + 4 + 4 = 13 channels
                    
                    try:
                        concatenated_latents = torch.cat(
                            [latent_model_input, mask_input, masked_image_latents, ref_latents], 
                            dim=1
                        )
                        print(f"🔍 UNet inputs - Concatenated: {concatenated_latents.shape}, Audio: {audio_tensor.shape}")
                        
                        # Validate expected shape
                        expected_channels = 13  # 4 + 1 + 4 + 4
                        if concatenated_latents.shape[1] != expected_channels:
                            raise ValueError(f"Expected {expected_channels} channels, got {concatenated_latents.shape[1]}")
                            
                    except Exception as concat_error:
                        print(f"❌ Concatenation failed: {concat_error}")
                        print("🔧 Attempting to fix tensor shapes...")
                        
                        # Emergency fix: ensure all tensors have compatible shapes
                        batch_size = latent_model_input.shape[0]
                        
                        # Ensure mask_input has correct shape
                        if mask_input.shape[1] != 1:
                            mask_input = mask_input[:, :1, ...]  # Take only first channel
                        
                        # Retry concatenation
                        concatenated_latents = torch.cat(
                            [latent_model_input, mask_input, masked_image_latents, ref_latents], 
                            dim=1
                        )
                        print(f"🔧 Fixed concatenated shape: {concatenated_latents.shape}")
                    
                    # ONNX inference with FP32 (matching ONNX model precision)
                    noise_pred = lip_syncer.run(None, {
                        'sample': concatenated_latents.cpu().numpy().astype(numpy.float32),  # FP32
                        'timesteps': numpy.array([timestep.cpu().numpy()], dtype=numpy.int64),
                        'encoder_hidden_states': audio_tensor.cpu().numpy().astype(numpy.float32)  # FP32
                    })[0]
                    
                    noise_pred = torch.from_numpy(noise_pred).to(target_device, dtype=torch.float32)
                    
                    # Apply CFG if enabled
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # Simple denoising step (without scheduler for simplicity)
                    latents = latents - 0.1 * noise_pred  # Simple update
                    
                    print(f"🔍 Final latents shape: {latents.shape}")
                    
                    # Clean up all remaining tensors before VAE decode
                    del audio_tensor, mask_input, masked_image_latents, ref_latents, noise_pred, concatenated_latents
                    torch.cuda.empty_cache()
                    
                    print(f"🔍 Final output_latent shape: {latents.shape}")
                    print(f"🔍 About to decode with VAE...")
                    
                    # 🏗️ STEP 6: VAE decode with ultra-conservative memory management
                    close_vision_frame = normalize_latentsync_frame_conservative(latents)
                    
                    print("🔍 Model output shape:", close_vision_frame.shape if close_vision_frame is not None else "None")
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"❌ OOM during LatentSync processing! Memory is critically low.", __name__)
                    torch.cuda.empty_cache()
                    # Return original frame as last resort
                    return close_vision_frame
                else:
                    logger.error(f"LatentSync processing failed: {str(e)}", __name__)
                    return close_vision_frame
            except Exception as e:
                logger.error(f"LatentSync processing failed: {str(e)}", __name__)
                return close_vision_frame
        else:
            # Wav2Lip-style direct inference
            print("🚀 Executing Wav2Lip path...")
            print(f"🔍 Wav2Lip input shapes - audio: {temp_audio_frame.shape}, vision: {close_vision_frame.shape}")
            
            close_vision_frame = lip_syncer.run(None, {
                'source': temp_audio_frame,
                'target': close_vision_frame
            })[0]
            
            print(f"🔍 Wav2Lip raw output shape: {close_vision_frame.shape}")
            close_vision_frame = normalize_close_frame(close_vision_frame)
            print(f"🔍 Wav2Lip normalized output shape: {close_vision_frame.shape}")

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


# Prepare audio for LatentSync: raw waveform → (1, N, 384)
# Input: raw audio waveform → Output: (1, N, 384)
def prepare_latentsync_audio(raw_audio_waveform: numpy.ndarray, sample_rate: int = 16000) -> torch.Tensor:
    """
    Convert a raw audio waveform into a torch.Tensor of Whisper encoder embeddings (LatentSync format).
    The output tensor has shape [1, N, 384] (batch_size=1, sequence_length=N audio tokens, embedding_dim=384),
    and uses torch.float16 dtype for compatibility with the LatentSync U-Net.
    
    :param raw_audio_waveform: NumPy array of audio samples (1D or 2D). 
                     If 2D (multi-channel), it will be converted to mono.
    :param sample_rate: Sample rate of the audio. LatentSync expects 16 kHz.
    :return: A float16 torch.Tensor of shape [1, N, 384] containing Whisper Tiny encoder embeddings.
    """
    try:
        import tempfile
        import os
        
        print(f"🔍 prepare_latentsync_audio input shape: {raw_audio_waveform.shape}")
        print(f"🔍 prepare_latentsync_audio input dtype: {raw_audio_waveform.dtype}")
        print(f"🔍 prepare_latentsync_audio sample_rate: {sample_rate}")
        
        # Validate input
        if raw_audio_waveform is None:
            raise ValueError("Audio waveform is None")
        if not isinstance(raw_audio_waveform, numpy.ndarray):
            raise TypeError(f"Expected numpy array, got {type(raw_audio_waveform)}")
        if raw_audio_waveform.size == 0:
            print("⚠️ Empty audio detected, creating minimal audio signal")
            # Create minimal 1-second audio at 16kHz
            raw_audio_waveform = numpy.zeros(16000, dtype=numpy.float32)
        
        # Ensure the audio is mono. If stereo or multi-channel, average the channels to get mono.
        if raw_audio_waveform.ndim > 1:
            raw_audio_waveform = numpy.mean(raw_audio_waveform, axis=0)
        
        # Convert waveform to float32 numpy array (Whisper expects float32 PCM in [-1.0, 1.0]).
        raw_audio_waveform = raw_audio_waveform.astype(numpy.float32)
        
        # Normalize the audio to [-1, 1] range if it's not already. 
        if numpy.max(numpy.abs(raw_audio_waveform)) > 1.0:
            # If data looks like int16 PCM, scale accordingly
            if raw_audio_waveform.dtype == numpy.int16 or numpy.max(numpy.abs(raw_audio_waveform)) > 32767:
                raw_audio_waveform = raw_audio_waveform / 32768.0
            else:
                # General normalization (in case of float data that's not yet in [-1,1])
                raw_audio_waveform = raw_audio_waveform / numpy.max(numpy.abs(raw_audio_waveform))
        
        # Ensure minimum length for Whisper processing (at least 0.1 seconds)
        min_samples = int(0.1 * 16000)  # 0.1 seconds at 16kHz
        if len(raw_audio_waveform) < min_samples:
            print(f"⚠️ Audio too short ({len(raw_audio_waveform)} samples), padding to {min_samples} samples")
            padded_audio = numpy.zeros(min_samples, dtype=numpy.float32)
            padded_audio[:len(raw_audio_waveform)] = raw_audio_waveform
            raw_audio_waveform = padded_audio
        
        # If sample rate is not 16000 Hz, resample the audio to 16000.
        if sample_rate != 16000:
            raw_audio_waveform = resample_audio(raw_audio_waveform, sample_rate, 16000)
            sample_rate = 16000
        
        print(f"🔍 Processed audio shape: {raw_audio_waveform.shape}")
        print(f"🔍 Processed audio min/max: {raw_audio_waveform.min():.4f}/{raw_audio_waveform.max():.4f}")
        
        # Create a temporary audio file since Audio2Feature expects a file path
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_audio_path = temp_file.name
        
        try:
            # Write the audio waveform to temporary file
            # Use librosa for better format support and compatibility
            try:
                # Try soundfile first (faster for WAV)
                sf.write(temp_audio_path, raw_audio_waveform, sample_rate)
                print(f"🔍 Wrote audio to {temp_audio_path}")
            except Exception as sf_error:
                # Fallback to librosa if soundfile fails
                print(f"⚠️ soundfile failed, using librosa: {sf_error}")
                # Use scipy.io.wavfile as another fallback
                from scipy.io import wavfile
                # Convert to int16 for wavfile
                audio_int16 = (raw_audio_waveform * 32767).astype(numpy.int16)
                wavfile.write(temp_audio_path, sample_rate, audio_int16)
                print(f"🔍 Wrote audio via scipy to {temp_audio_path}")
            
            # Use Audio2Feature to extract features from the audio file
            # This follows the same pattern as in lipsync_pipeline.py
            print("🔍 Calling audio2feat...")
            try:
                # Validate that the audio encoder is properly loaded with retry logic
                encoder = None
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        print(f"🔄 Audio encoder attempt {attempt + 1}/{max_retries}")
                        encoder = get_audio_encoder()
                        
                        if encoder is None:
                            raise RuntimeError("Audio encoder is None")
                        if not hasattr(encoder, 'audio2feat'):
                            raise RuntimeError("Audio encoder does not have audio2feat method")
                        if not hasattr(encoder, 'model') or encoder.model is None:
                            raise RuntimeError("Audio encoder model is None")
                        
                        # If we get here, encoder is valid
                        print(f"✅ Audio encoder validated on attempt {attempt + 1}")
                        break
                        
                    except Exception as encoder_error:
                        print(f"⚠️ Audio encoder attempt {attempt + 1} failed: {encoder_error}")
                        if attempt < max_retries - 1:
                            # Clear the global encoder and try again
                            global audio_encoder
                            audio_encoder = None
                            print("🔄 Clearing audio encoder for retry...")
                            # Small delay before retry
                            import time
                            time.sleep(1)
                        else:
                            # Final attempt failed
                            raise RuntimeError(f"Failed to initialize audio encoder after {max_retries} attempts: {encoder_error}")
                
                print(f"🔍 Audio encoder type: {type(encoder)}")
                print(f"🔍 Calling audio2feat on file: {temp_audio_path}")
                print(f"🔍 File exists: {os.path.exists(temp_audio_path)}")
                print(f"🔍 File size: {os.path.getsize(temp_audio_path) if os.path.exists(temp_audio_path) else 'N/A'} bytes")
                
                # Call audio2feat - encoder should already be on correct device
                audio_feat = encoder.audio2feat(temp_audio_path)
                
                print(f"🔍 Audio2feat returned shape: {audio_feat.shape}")
                print(f"🔍 Audio2feat returned dtype: {audio_feat.dtype}")
                print(f"🔍 Audio2feat returned device: {audio_feat.device}")
                
                # Validate the output
                if audio_feat is None:
                    raise ValueError("Audio2feat returned None")
                if not isinstance(audio_feat, torch.Tensor):
                    raise TypeError(f"Audio2feat returned {type(audio_feat)}, expected torch.Tensor")
                if audio_feat.numel() == 0:
                    raise ValueError("Audio2feat returned empty tensor")
                
                # audio_feat should be a tensor of shape (N, 384) where N is number of audio tokens
                # Add batch dimension to make it (1, N, 384)
                if audio_feat.ndim == 2:
                    audio_feat = audio_feat.unsqueeze(0)  # (N, 384) -> (1, N, 384)
                elif audio_feat.ndim == 3:
                    # Handle case where Audio2Feature returns [B, T, F] format
                    # Flatten batch and time dimensions: [B, T, F] -> [1, B*T, F]
                    B, T, F = audio_feat.shape
                    audio_feat = audio_feat.reshape(1, B * T, F)
                    print(f"🔍 Reshaped 3D audio_feat from [{B}, {T}, {F}] to {audio_feat.shape}")
                
                print(f"🔍 Final audio_feat shape: {audio_feat.shape}")
                
                # Validate output shape
                if audio_feat.shape[0] == 0 or audio_feat.shape[1] == 0:
                    raise ValueError(f"Invalid audio feature shape: {audio_feat.shape}")
                if audio_feat.shape[2] != 384:
                    raise ValueError(f"Expected 384 features, got {audio_feat.shape[2]}")
                
                # Keep on same device as audio encoder for consistency, ensure float32
                target_device = audio_feat.device
                audio_feat = audio_feat.to(target_device, dtype=torch.float32)
                
                # 🔧 CRITICAL: Ensure output is on same device as audio encoder
                encoder = get_audio_encoder()
                if hasattr(encoder, 'model') and encoder.model is not None:
                    encoder_device = encoder.model.device
                    if audio_feat.device != encoder_device:
                        print(f"🔧 Moving audio features from {audio_feat.device} to {encoder_device}")
                        audio_feat = audio_feat.to(encoder_device, dtype=torch.float32)
                
                # Ensure no mixed precision - force FP32 throughout
                if audio_feat.dtype != torch.float32:
                    print(f"🔧 Converting audio features from {audio_feat.dtype} to float32")
                    audio_feat = audio_feat.float()
                
                print(f"🔍 Returning audio tensor shape: {audio_feat.shape}, dtype: {audio_feat.dtype}, device: {audio_feat.device}")
                return audio_feat
                
            except Exception as audio_feat_error:
                print(f"❌ Audio2feat processing failed: {audio_feat_error}")
                print(f"❌ Error type: {type(audio_feat_error)}")
                import traceback
                print(f"❌ Traceback: {traceback.format_exc()}")
                
                # Try to estimate reasonable number of tokens based on input
                if hasattr(raw_audio_waveform, 'shape') and len(raw_audio_waveform.shape) > 0:
                    audio_samples = raw_audio_waveform.shape[0]
                    audio_duration = audio_samples / sample_rate
                    estimated_tokens = max(10, int(audio_duration * 50))  # ~50 tokens per second
                else:
                    estimated_tokens = 25  # Default fallback
                    
                # Use same device as audio encoder if available
                try:
                    encoder = get_audio_encoder()
                    if hasattr(encoder, 'model') and encoder.model is not None:
                        fallback_device = encoder.model.device
                    else:
                        fallback_device = "cuda" if torch.cuda.is_available() else "cpu"
                except:
                    fallback_device = "cuda" if torch.cuda.is_available() else "cpu"
                
                fallback_tensor = torch.zeros(1, estimated_tokens, 384, dtype=torch.float32, device=fallback_device)
                print(f"🔧 Fallback tensor shape: {fallback_tensor.shape}, dtype: {fallback_tensor.dtype}, device: {fallback_tensor.device}")
                return fallback_tensor
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    except Exception as e:
        print(f"❌ prepare_latentsync_audio failed: {str(e)}")
        print(f"❌ Input was: {type(raw_audio_waveform)}, shape: {getattr(raw_audio_waveform, 'shape', 'no shape')}")
        # Return a more reasonable fallback audio tensor based on input
        print("🔧 Creating fallback audio tensor...")
        
        # Try to estimate reasonable number of tokens based on input
        if hasattr(raw_audio_waveform, 'shape') and len(raw_audio_waveform.shape) > 0:
            audio_samples = raw_audio_waveform.shape[0]
            audio_duration = audio_samples / sample_rate
            estimated_tokens = max(10, int(audio_duration * 50))  # ~50 tokens per second
        else:
            estimated_tokens = 25  # Default fallback
            
        # Use same device as audio encoder if available
        try:
            encoder = get_audio_encoder()
            if hasattr(encoder, 'model') and encoder.model is not None:
                fallback_device = encoder.model.device
            else:
                fallback_device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            fallback_device = "cuda" if torch.cuda.is_available() else "cpu"
            
        fallback_tensor = torch.zeros(1, estimated_tokens, 384, dtype=torch.float32, device=fallback_device)
        print(f"🔧 Fallback tensor shape: {fallback_tensor.shape}, dtype: {fallback_tensor.dtype}, device: {fallback_tensor.device}")
        return fallback_tensor


# Input: (H, W, 3) BGR image (OpenCV), Output: torch.Tensor (1, 4, 64, 64)
def prepare_latentsync_frame(vision_frame: VisionFrame) -> torch.Tensor:
    """
    Converts a single BGR image (OpenCV format) to VAE latent representation
    for LatentSync. Output shape: (1, 4, 64, 64)
    """
    if vision_frame is None:
        raise ValueError("❌ vision_frame is None.")
    if not isinstance(vision_frame, numpy.ndarray):
        raise TypeError("❌ vision_frame is not a numpy array.")
    if vision_frame.size == 0:
        raise ValueError("❌ vision_frame is empty.")

    try:
        # 🛡️ Ensure valid OpenCV input: uint8 in [0, 255]
        if vision_frame.dtype != numpy.uint8:
            print(f"⚠️ Warning: vision_frame dtype is {vision_frame.dtype}, converting to uint8.")
            vision_frame = numpy.clip(vision_frame, 0, 255).astype(numpy.uint8)

        # ✅ Convert from BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(vision_frame, cv2.COLOR_BGR2RGB)

        # ✅ Resize to 512x512 (standard for VAE encoding)
        resized = cv2.resize(frame_rgb, (512, 512))
        print("✅ Resized frame to 512x512")

        # ✅ Normalize to [-1, 1] in float32 then keep as float32
        normalized = resized.astype(numpy.float32) / 255.0
        normalized = (normalized * 2.0) - 1.0

        # ✅ Change shape to (1, 3, 512, 512)
        tensor = torch.from_numpy(numpy.transpose(normalized, (2, 0, 1))).unsqueeze(0)

        # ✅ Encode with VAE to get latent representation
        vae_instance = get_vae()
        
        # 🔧 CRITICAL: Ensure device consistency between tensor and VAE
        vae_device = vae_instance.device
        print(f"🔧 VAE is on device: {vae_device}")
        
        # Move tensor to same device as VAE and ensure FP32
        tensor = tensor.to(vae_device, dtype=torch.float32)
        print(f"🔧 Input tensor moved to {vae_device} with dtype: {tensor.dtype}")
        
        # Ensure VAE is in FP32 precision for stability
        vae_instance = vae_instance.float()
        print(f"🔧 VAE set to FP32 precision on {vae_device}")
        
        with torch.no_grad():
            try:
                print(f"🔍 About to encode tensor shape: {tensor.shape}")
                encode_result = vae_instance.encode(tensor)
                print(f"🔍 Encode result type: {type(encode_result)}")
                print(f"🔍 Encode result latent_dist: {encode_result.latent_dist}")
                
                latent = encode_result.latent_dist.sample()
                
            except RuntimeError as encode_error:
                print(f"❌ VAE encoding failed: {encode_error}")
                # Try CPU fallback with proper device consistency
                print("🔄 Trying VAE encoding on CPU as fallback...")
                try:
                    vae_instance = vae_instance.cpu().float()  # Move VAE to CPU
                    tensor = tensor.cpu().to(torch.float32)    # Move tensor to CPU
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    encode_result = vae_instance.encode(tensor)
                    latent = encode_result.latent_dist.sample()
                    print("✅ CPU fallback successful")
                except Exception as cpu_error:
                    raise RuntimeError(f"VAE encoding failed on both GPU and CPU: GPU={encode_error}, CPU={cpu_error}")
        
        print(f"🔍 Latent after sampling: {latent}")
        print(f"🔍 Latent type: {type(latent)}")
        print(f"🔍 Latent shape: {latent.shape if latent is not None else 'None'}")
        
        # Handle missing VAE config attributes with defaults
        vae_config = vae_instance.config
        print(f"🔍 VAE config type: {type(vae_config)}")
        print(f"🔍 VAE config dir: {dir(vae_config)}")
        
        shift_factor = getattr(vae_config, 'shift_factor', 0.0)
        scaling_factor = getattr(vae_config, 'scaling_factor', 0.18215)
        
        # Handle None values explicitly
        if shift_factor is None:
            shift_factor = 0.0
        if scaling_factor is None:
            scaling_factor = 0.18215
        
        print(f"🔍 shift_factor: {shift_factor}")
        print(f"🔍 scaling_factor: {scaling_factor}")
        
        # Correct scaling: (latents - shift_factor) * scaling_factor
        latent = (latent - shift_factor) * scaling_factor
        latent = latent.to(torch.float32)  # Ensure FP32 precision
        
        # 🔧 CRITICAL: Ensure output latent is on same device as VAE
        latent = latent.to(vae_instance.device, dtype=torch.float32)
        
        print("🔍 VAE latent shape:", latent.shape)
        print("🔍 VAE latent dtype:", latent.dtype)
        print("🔍 VAE latent device:", latent.device)
        print("🔍 VAE latent - min/max:", latent.min().item(), latent.max().item())
        print("🔍 VAE latent - mean:", latent.mean().item())

        return latent  # Shape: (1, 4, 64, 64)

    except Exception as e:
        raise RuntimeError(f"❌ Failed to prepare vision frame: {str(e)}")


# Ultra-conservative VAE decode that immediately offloads VAE after use
def normalize_latentsync_frame_conservative(latent: torch.Tensor) -> VisionFrame:
    """
    Ultra-conservative VAE decode that immediately moves VAE back to CPU after use
    to prevent OOM issues in memory-constrained environments.
    """
    global vae  # Move global declaration to top
    
    if not isinstance(latent, torch.Tensor):
        raise TypeError("Input must be a torch tensor")
    
    try:
        print(f"🔍 Conservative decode input shape: {latent.shape}")
        
        # Handle different input shapes
        if latent.ndim == 5:
            pass  # (1, 4, 1, 64, 64) - keep as is
        elif latent.ndim == 4:
            latent = latent.unsqueeze(2)  # → (1, 4, 1, 64, 64)
        elif latent.ndim == 3:
            latent = latent.unsqueeze(0).unsqueeze(2)  # → (1, 4, 1, 64, 64)
        else:
            raise ValueError(f"Unexpected latent shape: {latent.shape}")
        
        # Ensure we have 4 channels
        if latent.shape[1] != 4:
            if latent.shape[1] >= 4:
                latent = latent[:, :4, ...]  # Take first 4 channels
            else:
                raise ValueError(f"Not enough channels: expected 4, got {latent.shape[1]}")

        with torch.no_grad():
            # 🔧 CRITICAL: Conservative GPU usage with FP32 for stability
            vae_instance = get_vae()  # This will load it if needed
            
            # Use same device as VAE with float32 precision
            latent = latent.to(vae_instance.device, dtype=torch.float32)
            print(f"🔧 VAE decode using {vae_instance.device} with float32 precision")
            
            # Apply VAE scaling
            vae_config = vae_instance.config
            shift_factor = getattr(vae_config, 'shift_factor', 0.0) or 0.0
            scaling_factor = getattr(vae_config, 'scaling_factor', 0.18215) or 0.18215
            
            latents = latent / scaling_factor + shift_factor
            latents = latents.to(torch.float32)  # Ensure FP32 precision
            
            # Reshape for VAE decode
            from einops import rearrange
            latents = rearrange(latents, "b c f h w -> (b f) c h w")  # (1, 4, 1, 64, 64) → (1, 4, 64, 64)
            
            print(f"🔍 About to VAE decode shape: {latents.shape}")
            
            # VAE decode with proper error handling
            try:
                decoded_latents = vae_instance.decode(latents).sample  # (1, 4, 64, 64) → (1, 3, 512, 512)
                print(f"✅ VAE decode completed on {vae_instance.device}")
            except Exception as decode_error:
                print(f"⚠️ VAE decode failed on {vae_instance.device}: {decode_error}")
                # Fallback to CPU if GPU fails
                if vae_instance.device.type == 'cuda':
                    print("🔄 Falling back to CPU for VAE decode")
                    vae_instance = vae_instance.cpu().float()  # Ensure FP32
                    latents = latents.cpu().to(torch.float32)  # Ensure FP32
                    decoded_latents = vae_instance.decode(latents).sample
                    print("✅ VAE decode completed on CPU (fallback)")
                else:
                    raise decode_error

        # Handle unexpected channel count
        if decoded_latents.shape[1] == 6:
            print("⚠️ VAE returned 6 channels, extracting first 3 (RGB)")
            decoded_latents = decoded_latents[:, :3, :, :]
        elif decoded_latents.shape[1] != 3:
            print(f"⚠️ Unexpected channel count: {decoded_latents.shape[1]}, expected 3")
            if decoded_latents.shape[1] > 3:
                decoded_latents = decoded_latents[:, :3, :, :]
            else:
                raise ValueError(f"Not enough channels: got {decoded_latents.shape[1]}, need 3")

        # Convert to image format
        pixel_values = rearrange(decoded_latents, "f c h w -> f h w c")  # (1, 3, 512, 512) → (1, 512, 512, 3)
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        image = images[0].cpu().numpy()  # (512, 512, 3)
        
        # Convert RGB → BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        print(f"🔍 Final image shape: {image_bgr.shape}")
        
        return image_bgr

    except Exception as e:
        # Emergency cleanup
        if vae is not None:
            try:
                vae = vae.cpu()
                torch.cuda.empty_cache()
            except:
                pass
        raise RuntimeError(f"❌ Conservative decode failed: {str(e)}")


def get_reference_frame(source_face : Face, target_face : Face, temp_vision_frame : VisionFrame) -> VisionFrame:
	pass


def process_frame(inputs : LipSyncerInputs) -> VisionFrame:
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
		# Clear GPU cache without moving models to CPU (keep device consistency)
		try:
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				torch.cuda.synchronize()
			print("🧹 GPU cache cleared after LatentSync frame processing")
		except Exception as cleanup_error:
			print(f"⚠️ Cleanup warning: {cleanup_error}")
	
	return target_vision_frame


def process_frames(source_paths : List[str], queue_payloads : List[QueuePayload], update_progress : UpdateProgress) -> None:
	reference_faces = get_reference_faces() if 'reference' in state_manager.get_item('face_selector_mode') else None
	source_audio_path = get_first(filter_audio_paths(source_paths))
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
	model_name = state_manager.get_item('lip_syncer_model')

	for queue_payload in process_manager.manage(queue_payloads):
		frame_number = queue_payload.get('frame_number')
		target_vision_path = queue_payload.get('frame_path')
		
		# Get appropriate audio frame based on model
		if model_name == 'latentsync':
			source_audio_frame = get_raw_audio_frame(source_audio_path, temp_video_fps, frame_number)
			if source_audio_frame is None or not numpy.any(source_audio_frame):
				# Create empty raw audio frame with consistent FP32 format
				source_audio_frame = create_empty_raw_audio_frame(temp_video_fps, sample_rate=16000)
		else:
			source_audio_frame = get_voice_frame(source_audio_path, temp_video_fps, frame_number)
			if not numpy.any(source_audio_frame):
				source_audio_frame = create_empty_audio_frame()
		
		target_vision_frame = read_image(target_vision_path)
		output_vision_frame = process_frame(
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
	output_vision_frame = process_frame(
	{
		'reference_faces': reference_faces,
		'source_audio_frame': source_audio_frame,
		'target_vision_frame': target_vision_frame
	})
	write_image(output_path, output_vision_frame)


def process_video(source_paths : List[str], temp_frame_paths : List[str]) -> None:
	source_audio_paths = filter_audio_paths(state_manager.get_item('source_paths'))
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
	for source_audio_path in source_audio_paths:
		read_static_voice(source_audio_path, temp_video_fps)
	processors.multi_process_frames(source_paths, temp_frame_paths, process_frames)
