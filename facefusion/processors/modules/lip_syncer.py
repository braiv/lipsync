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
from facefusion.audio import create_empty_audio_frame, get_voice_frame, read_static_voice, get_raw_audio_frame
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
from diffusers import DDIMScheduler

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

def get_audio_encoder():
    """Lazy loading of Whisper audio encoder"""
    global audio_encoder
    if audio_encoder is None:
        if not LATENTSYNC_AVAILABLE:
            raise RuntimeError("LatentSync is not available. Cannot load audio encoder.")
            
        print("🎵 Loading Whisper Tiny encoder...")
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            available_gb = available_memory / 1024**3
            print(f"💾 Available memory before audio encoder: {available_gb:.1f} GB")
            
            # Optimized for T4 16GB - only use CPU if very low memory
            if available_gb < 1.0:
                print("⚠️ Very low memory! Loading audio encoder on CPU.")
                audio_device = "cpu"
            else:
                audio_device = device
        else:
            audio_device = device
            
        audio_encoder = Audio2Feature(model_path="checkpoints/whisper/tiny.pt", device=audio_device)
        print("✅ Audio encoder loaded.")
    return audio_encoder

def get_vae():
    """Lazy loading of VAE model"""
    global vae
    if vae is None:
        print("🖼️ Loading VAE model...")
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            available_gb = available_memory / 1024**3
            print(f"💾 Available memory before VAE: {available_gb:.1f} GB")
            
            # Optimized for T4 16GB - only use CPU if very low memory
            if available_gb < 2.0:
                print("⚠️ Low memory! Loading VAE on CPU.")
                vae_device = "cpu"
            else:
                vae_device = "cuda"
        else:
            vae_device = "cpu"
            
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(vae_device).half().eval()
        print("✅ VAE loaded.")
    return vae

def get_projection_weight():
    """Lazy loading of projection weight"""
    global projection_weight
    if projection_weight is None:
        projection_weight = torch.randn(384, 4).half().to("cuda" if torch.cuda.is_available() else "cpu")
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
    lip_syncer = get_inference_pool().get('lip_syncer')  # ONNX Runtime session
    model_name = state_manager.get_item('lip_syncer_model')
    
    print(f"🔍 Forward function called with model: {model_name}")
    print(f"🔍 Input close_vision_frame shape: {close_vision_frame.shape}")
    print(f"🔍 Input close_vision_frame dtype: {close_vision_frame.dtype}")

    with conditional_thread_semaphore():
        if model_name == 'latentsync':
            if not LATENTSYNC_AVAILABLE:
                logger.error("LatentSync model selected but LatentSync is not available!", __name__)
                return close_vision_frame
                
            print("🚀 Executing LatentSync path...")
            try:
                with torch.no_grad():
                    # 🧹 Initial memory cleanup
                    torch.cuda.empty_cache()
                    
                    # 💾 Check available memory and adjust settings
                    if torch.cuda.is_available():
                        available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                        available_gb = available_memory / 1024**3
                        print(f"💾 Available GPU memory: {available_gb:.1f} GB")
                        
                        # 🔥 MEMORY OPTIMIZATION: Optimized thresholds for different VRAM sizes
                        if available_gb < 3.0:
                            print("⚠️ Low memory detected! Disabling CFG to prevent OOM.")
                            guidance_scale = 1.0  # Disable CFG
                            num_inference_steps = 10  # Reduce steps
                        elif available_gb < 6.0:
                            print("💡 Medium memory detected. Using standard CFG.")
                            guidance_scale = 1.5  # Standard CFG
                            num_inference_steps = 15  # Slightly reduced steps
                        else:
                            print("🚀 High memory detected! Using optimal settings.")
                            guidance_scale = 1.5  # Full CFG
                            num_inference_steps = 20  # Full quality
                    else:
                        guidance_scale = 1.5
                        num_inference_steps = 20
                    
                    do_classifier_free_guidance = guidance_scale > 1.0
                    
                    # Prepare audio input -> (1, N, 384) where N depends on audio length
                    # LatentSync uses 16kHz audio for Whisper, not 48kHz
                    audio_tensor = prepare_latentsync_audio(temp_audio_frame, sample_rate=16000)
                    
                    # Apply classifier-free guidance for audio
                    if do_classifier_free_guidance:
                        # Create unconditional audio embeddings (zeros)
                        null_audio_embeds = torch.zeros_like(audio_tensor)
                        # Concatenate with conditional embeddings
                        audio_tensor = torch.cat([null_audio_embeds, audio_tensor])
                                         
                    # Prepare video input -> (1, 4, 64, 64)
                    video_latent = prepare_latentsync_frame(close_vision_frame)

                    # 1. Create initial noise latents for diffusion
                    # Shape: (1, 4, 1, 64, 64) for single frame
                    noise = torch.randn(1, 4, 1, 64, 64, dtype=torch.float16, device=device)
                    
                    # 2. Create proper mask for mouth region (binary mask)
                    mask_height, mask_width = 64, 64
                    mouth_mask = torch.zeros((mask_height, mask_width), dtype=torch.float16, device=device)
                    # Create mouth region (lower center part of face)
                    mouth_y_start = int(mask_height * 0.6)  # Lower 40% of face
                    mouth_y_end = int(mask_height * 0.9)
                    mouth_x_start = int(mask_width * 0.3)   # Center 40% width
                    mouth_x_end = int(mask_width * 0.7)
                    mouth_mask[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end] = 1.0
                    
                    # Shape: (1, 1, 1, 64, 64) - single channel mask
                    mask_latents = mouth_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    
                    # Duplicate masks for classifier-free guidance if needed
                    if do_classifier_free_guidance:
                        mask_latents = torch.cat([mask_latents] * 2)
                    
                    # 3. Create masked image latents
                    # Apply mask to video latents (mask out mouth region)
                    masked_image_latents = video_latent * (1 - mask_latents[:1, 0:1])  # Use first mask, expand to 4 channels
                    masked_image_latents = masked_image_latents.unsqueeze(2)  # Add temporal dimension
                    
                    # Duplicate masked image latents for classifier-free guidance if needed
                    if do_classifier_free_guidance:
                        masked_image_latents = torch.cat([masked_image_latents] * 2)
                    
                    # Add temporal dimension to video_latent for reference
                    ref_latents = video_latent.unsqueeze(2)
                    
                    # Duplicate reference latents for classifier-free guidance if needed
                    if do_classifier_free_guidance:
                        ref_latents = torch.cat([ref_latents] * 2)
                    
                    # 🧹 Clean up intermediate tensors
                    del video_latent, mouth_mask
                    torch.cuda.empty_cache()
                    
                    # 4. Setup proper DDIM scheduler (following lipsync_pipeline.py and inference.py)
                    # Use actual DDIM scheduler instead of manual creation
                    scheduler = DDIMScheduler(
                        num_train_timesteps=1000,
                        beta_start=0.00085,
                        beta_end=0.012,
                        beta_schedule="linear",
                        clip_sample=False,
                        set_alpha_to_one=False,
                        steps_offset=1,
                        prediction_type="epsilon"
                    )
                    
                    # Set timesteps for inference (following lipsync_pipeline.py)
                    scheduler.set_timesteps(num_inference_steps, device=device)
                    timesteps = scheduler.timesteps
                    
                    # 5. Initialize latents with noise (following lipsync_pipeline.py: prepare_latents)
                    # Scale initial noise by scheduler.init_noise_sigma (standard DDIM initialization)
                    latents = noise * scheduler.init_noise_sigma
                    del noise  # Clean up noise tensor
                    
                    # 6. Denoising loop (following lipsync_pipeline.py structure)
                    for i, t in enumerate(timesteps):
                        # 🧹 Memory cleanup every few iterations
                        if i % 5 == 0:
                            torch.cuda.empty_cache()
                        
                        # Expand latents for classifier-free guidance
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        
                        # Scale model input (following lipsync_pipeline.py: scheduler.scale_model_input)
                        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                        
                        # Concatenate all inputs: [latents, mask, masked_image, ref]
                        # Following train_unet.py line 350: torch.cat([noisy_gt_latents, masks, masked_latents, ref_latents], dim=1)
                        concatenated_latents = torch.cat(
                            [latent_model_input, mask_latents, masked_image_latents, ref_latents], 
                            dim=1
                        )
                        
                        # 🧹 Clean up intermediate tensor
                        del latent_model_input
                        
                        # Run ONNX inference for this timestep
                        noise_pred = lip_syncer.run(None, {
                            'sample': concatenated_latents.cpu().numpy().astype(numpy.float16),
                            'timesteps': numpy.array([t.cpu().numpy()], dtype=numpy.int64),
                            'encoder_hidden_states': audio_tensor.cpu().numpy().astype(numpy.float16)
                        })[0]
                        
                        # 🔍 Debug ONNX output shape
                        if i == 0:  # Only print on first iteration
                            print(f"🔍 ONNX noise_pred shape: {noise_pred.shape}")
                            print(f"🔍 ONNX noise_pred dtype: {noise_pred.dtype}")
                        
                        # 🧹 Clean up concatenated tensor immediately
                        del concatenated_latents
                        
                        # Convert back to torch tensor
                        noise_pred = torch.from_numpy(noise_pred).to(device)
                        
                        # Perform guidance if needed (following lipsync_pipeline.py line 450-452)
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                            # Clean up intermediate tensors
                            del noise_pred_uncond, noise_pred_cond
                        
                        # DDIM update using scheduler.step() (following lipsync_pipeline.py)
                        latents = scheduler.step(noise_pred, t, latents).prev_sample
                        
                        # 🧹 Clean up noise prediction
                        del noise_pred
                    
                    # Final output is the denoised latents
                    output_latent = latents
                    
                    # 🔍 Debug final output shape
                    print(f"🔍 Final output_latent shape: {output_latent.shape}")
                    print(f"🔍 Final output_latent dtype: {output_latent.dtype}")
                    print(f"🔍 Final output_latent min/max: {output_latent.min().item():.4f}/{output_latent.max().item():.4f}")
                    
                    # 🧹 Clean up all intermediate tensors
                    del latents, mask_latents, masked_image_latents, ref_latents, audio_tensor
                    torch.cuda.empty_cache()

                    if output_latent is None:
                        raise RuntimeError("ONNX inference returned None.")

                    # Convert numpy array to torch tensor if needed
                    if isinstance(output_latent, numpy.ndarray):
                        output_latent = torch.from_numpy(output_latent).to(torch.float16).to(device)

                    # Convert Input: (1, 4, 1, 64, 64) to Output: (512, 512, 3) for downstream transpose
                    print(f"🔍 About to call normalize_latentsync_frame with shape: {output_latent.shape}")
                    close_vision_frame = normalize_latentsync_frame(output_latent)
                    
                    # 🧹 Final cleanup
                    del output_latent
                    torch.cuda.empty_cache()

                    # After model inference
                    print("🔍 Model output shape:", close_vision_frame.shape if close_vision_frame is not None else "None")
                    print("🔍 Model output dtype:", close_vision_frame.dtype if close_vision_frame is not None else "None")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"❌ OOM during LatentSync processing! Try reducing video resolution or disabling CFG.", __name__)
                    # 🧹 Emergency cleanup
                    torch.cuda.empty_cache()
                    return close_vision_frame
                else:
                    logger.error(f"LatentSync processing failed: {str(e)}", __name__)
                    return close_vision_frame
            except Exception as e:
                logger.error(f"LatentSync processing failed: {str(e)}", __name__)
                return close_vision_frame
            finally:
                # 🧹 Comprehensive cleanup
                for var in ['audio_tensor', 'video_latent', 'noise_pred', 'output_latent', 'latents', 
                           'mask_latents', 'masked_image_latents', 'ref_latents', 'concatenated_latents',
                           'latent_model_input', 'noise', 'mouth_mask']:
                    if var in locals():
                        del locals()[var]
                torch.cuda.empty_cache()
        else:
            # Wav2Lip-style direct inference with image and mel-spectrogram
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
        
        # If sample rate is not 16000 Hz, resample the audio to 16000.
        if sample_rate != 16000:
            raw_audio_waveform = resample_audio(raw_audio_waveform, sample_rate, 16000)
            sample_rate = 16000
        
        # Create a temporary audio file since Audio2Feature expects a file path
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_audio_path = temp_file.name
        
        try:
            # Write the audio waveform to temporary file
            # Use librosa for better format support and compatibility
            try:
                # Try soundfile first (faster for WAV)
                sf.write(temp_audio_path, raw_audio_waveform, sample_rate)
            except Exception as sf_error:
                # Fallback to librosa if soundfile fails
                print(f"⚠️ soundfile failed, using librosa: {sf_error}")
                # Use scipy.io.wavfile as another fallback
                from scipy.io import wavfile
                # Convert to int16 for wavfile
                audio_int16 = (raw_audio_waveform * 32767).astype(numpy.int16)
                wavfile.write(temp_audio_path, sample_rate, audio_int16)
            
            # Use Audio2Feature to extract features from the audio file
            # This follows the same pattern as in lipsync_pipeline.py
            audio_feat = get_audio_encoder().audio2feat(temp_audio_path)
            
            # audio_feat should be a tensor of shape (N, 384) where N is number of audio tokens
            # Add batch dimension to make it (1, N, 384)
            if audio_feat.ndim == 2:
                audio_feat = audio_feat.unsqueeze(0)  # (N, 384) -> (1, N, 384)
            
            # Convert to float16 and move to the correct device
            audio_feat = audio_feat.to(device, dtype=torch.float16)
            
            return audio_feat
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    except Exception as e:
        raise RuntimeError(f"❌ Failed to prepare LatentSync audio: {str(e)}")


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

        # ✅ Normalize to [-1, 1] in float32 then convert to float16
        normalized = resized.astype(numpy.float32) / 255.0
        normalized = (normalized * 2.0) - 1.0
        normalized = normalized.astype(numpy.float16)

        # ✅ Change shape to (1, 3, 512, 512)
        tensor = torch.from_numpy(numpy.transpose(normalized, (2, 0, 1))).unsqueeze(0)

        # ✅ Move to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = tensor.to(device).to(torch.float16)

        # ✅ Encode with VAE to get latent representation
        with torch.no_grad():
            # Following train_unet.py and lipsync_pipeline.py VAE scaling
            latent = get_vae().encode(tensor).latent_dist.sample()
            # Correct scaling: (latents - shift_factor) * scaling_factor
            latent = (latent - get_vae().config.shift_factor) * get_vae().config.scaling_factor
            latent = latent.to(torch.float16)
            
            print("🔍 VAE latent shape:", latent.shape)
            print("🔍 VAE latent - min/max:", latent.min().item(), latent.max().item())
            print("🔍 VAE latent - mean:", latent.mean().item())

        return latent  # Shape: (1, 4, 64, 64)

    except Exception as e:
        raise RuntimeError(f"❌ Failed to prepare vision frame: {str(e)}")


# Convert LatentSync UNet output latent back to displayable image (512x512x3 RGB)
# Input: (1, 4, 1, 64, 64) → Output: (512, 512, 3) BGR for OpenCV
def normalize_latentsync_frame(latent: torch.Tensor) -> VisionFrame:
    if not isinstance(latent, torch.Tensor):
        raise TypeError("Input must be a torch tensor")
    
    try:
        print(f"🔍 normalize_latentsync_frame input shape: {latent.shape}")
        print(f"🔍 normalize_latentsync_frame input dtype: {latent.dtype}")
        
        # Handle different input shapes - following lipsync_pipeline.py decode_latents
        if latent.ndim == 5:
            # Input: (1, 4, 1, 64, 64) - keep as is for rearrange
            pass
        elif latent.ndim == 4:
            # Input: (1, 4, 64, 64) - add temporal dimension
            latent = latent.unsqueeze(2)  # → (1, 4, 1, 64, 64)
        elif latent.ndim == 3:
            # Input: (4, 64, 64) - add batch and temporal dimensions
            latent = latent.unsqueeze(0).unsqueeze(2)  # → (1, 4, 1, 64, 64)
        else:
            raise ValueError(f"Unexpected latent shape: {latent.shape}")
        
        # Check if we have the expected number of channels (4 for VAE latents)
        if latent.shape[1] != 4:
            print(f"⚠️ Warning: Expected 4 channels, got {latent.shape[1]}. Attempting to extract first 4 channels.")
            if latent.shape[1] >= 4:
                latent = latent[:, :4, ...]  # Take first 4 channels
            else:
                raise ValueError(f"Not enough channels: expected 4, got {latent.shape[1]}")

        with torch.no_grad():
            print("🔍 Before VAE decode - latent shape:", latent.shape)
            print("🔍 Before VAE decode - latent min/max:", latent.min().item(), latent.max().item())

            latent = latent.to(torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Following lipsync_pipeline.py decode_latents method exactly:
            # Step 1: Apply VAE scaling (line 142)
            latents = latent / get_vae().config.scaling_factor + get_vae().config.shift_factor
            
            # Step 2: Reshape for VAE decode (line 143)
            from einops import rearrange
            latents = rearrange(latents, "b c f h w -> (b f) c h w")  # (1, 4, 1, 64, 64) → (1, 4, 64, 64)
            
            # Step 3: VAE decode (line 144)
            decoded_latents = get_vae().decode(latents).sample  # (1, 4, 64, 64) → (1, 3, 512, 512)

            print("🔍 After VAE decode - raw shape:", decoded_latents.shape)
            print("🔍 After VAE decode - raw min/max:", decoded_latents.min().item(), decoded_latents.max().item())
            
            # 🛠️ Handle unexpected channel count from VAE
            if decoded_latents.shape[1] == 6:
                print("⚠️ VAE returned 6 channels, extracting first 3 (RGB)")
                decoded_latents = decoded_latents[:, :3, :, :]  # Take first 3 channels
                print("🔍 After channel extraction - shape:", decoded_latents.shape)
            elif decoded_latents.shape[1] != 3:
                print(f"⚠️ Unexpected channel count: {decoded_latents.shape[1]}, expected 3")
                # If we have more than 3 channels, take the first 3
                if decoded_latents.shape[1] > 3:
                    decoded_latents = decoded_latents[:, :3, :, :]
                else:
                    raise ValueError(f"Not enough channels: got {decoded_latents.shape[1]}, need 3")

            del latent, latents
            torch.cuda.empty_cache()

            # Following lipsync_pipeline.py pixel_values_to_images method (line 254-257):
            # Step 1: Rearrange dimensions
            pixel_values = rearrange(decoded_latents, "f c h w -> f h w c")  # (1, 3, 512, 512) → (1, 512, 512, 3)
            
            # Step 2: Normalize to [0, 1]
            pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
            
            # Step 3: Convert to uint8
            images = (pixel_values * 255).to(torch.uint8)
            
            # Step 4: Convert to numpy and remove batch dimension
            image = images[0].cpu().numpy()  # (512, 512, 3)
            
            print("🔍 After normalization - shape:", image.shape)
            print("🔍 After normalization - min/max:", image.min(), image.max())
            
            # Convert RGB → BGR for OpenCV compatibility
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            print("🧪 Final normalized frame shape:", image_bgr.shape)
            print("🧪 Final normalized frame - dtype:", image_bgr.dtype)

        return image_bgr  # (512, 512, 3) BGR uint8

    except Exception as e:
        raise RuntimeError(f"❌ Failed to normalize frame: {str(e)}")


def get_reference_frame(source_face : Face, target_face : Face, temp_vision_frame : VisionFrame) -> VisionFrame:
	pass


def process_frame(inputs : LipSyncerInputs) -> VisionFrame:
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
				# Create empty raw audio frame (16kHz for Whisper)
				frame_duration_samples = int(16000 / temp_video_fps)
				source_audio_frame = numpy.zeros(frame_duration_samples, dtype=numpy.float32)
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
		# Create empty raw audio frame (1 second at 16kHz for Whisper)
		source_audio_frame = numpy.zeros(16000, dtype=numpy.float32)
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
