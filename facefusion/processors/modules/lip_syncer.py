from argparse import ArgumentParser
from functools import lru_cache
from typing import List

import cv2
import numpy
import torch
import torch.nn.functional as F

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
import facefusion.processors.core as processors
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, inference_manager, logger, process_manager, state_manager, voice_extractor, wording
from facefusion.audio import create_empty_audio_frame, get_voice_frame, read_static_voice
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


# Load VAE (Stable Diffusion 1.5 compatible)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to("cuda").half().eval()

# Linear projection layer: projects from 4 → 384
projection_weight = torch.randn(384, 4).half().to("cuda" if torch.cuda.is_available() else "cpu")  # or load if saved

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


def sync_lip(target_face : Face, temp_audio_frame : AudioFrame, temp_vision_frame : VisionFrame) -> VisionFrame:
	model_size = get_model_options().get('size')
	temp_audio_frame = prepare_audio_frame(temp_audio_frame)
	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark_set.get('5/68'), 'ffhq_512', (512, 512))
	face_landmark_68 = cv2.transform(target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
	bounding_box = create_bounding_box(face_landmark_68)
	bounding_box[1] -= numpy.abs(bounding_box[3] - bounding_box[1]) * 0.125
	mouth_mask = create_mouth_mask(face_landmark_68)
	box_mask = create_static_box_mask(crop_vision_frame.shape[:2][::-1], state_manager.get_item('face_mask_blur'), state_manager.get_item('face_mask_padding'))
	crop_masks =\
	[
		mouth_mask,
		box_mask
	]

	if 'occlusion' in state_manager.get_item('face_mask_types'):
		occlusion_mask = create_occlusion_mask(crop_vision_frame)
		crop_masks.append(occlusion_mask)

	close_vision_frame, close_matrix = warp_face_by_bounding_box(crop_vision_frame, bounding_box, model_size)
	close_vision_frame = prepare_crop_frame(close_vision_frame)
	close_vision_frame = forward(temp_audio_frame, close_vision_frame)
	#close_vision_frame = normalize_close_frame(close_vision_frame)
	crop_vision_frame = cv2.warpAffine(close_vision_frame, cv2.invertAffineTransform(close_matrix), (512, 512), borderMode = cv2.BORDER_REPLICATE)
	crop_mask = numpy.minimum.reduce(crop_masks)
	print("🔍 crop_mask min/max:", crop_mask.min(), crop_mask.max())

	paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
	return paste_vision_frame


def forward(temp_audio_frame: AudioFrame, close_vision_frame: VisionFrame) -> VisionFrame:
    lip_syncer = get_inference_pool().get('lip_syncer')  # ONNX Runtime session
    model_name = state_manager.get_item('lip_syncer_model')

    with conditional_thread_semaphore():
        if model_name == 'latentsync':
            try:
                with torch.no_grad():
                    # Prepare audio input → (1, 13, 8, 64, 64)
                    audio_tensor = prepare_latentsync_audio(temp_audio_frame)

                    # Prepare video latent → (1, 4, 64, 64)
                    vision_latent = prepare_latentsync_frame(close_vision_frame)

					# Reorders the dimension from (1, 4, 64, 64) to (1, 64, 64, 4). Then, flattens (64 x 64 = 4096) it to (1, 4096, 4)
                    encoder_hidden_states = vision_latent.permute(0, 2, 3, 1).reshape(1, -1, 4)  # (1, T*H*W, 4)
                    encoder_hidden_states = torch.nn.functional.linear(encoder_hidden_states, projection_weight)  # Project to 384-dim (1, 4096, 384)
                    encoder_hidden_states = encoder_hidden_states.cpu().numpy()

                    #print("Audio tensor shape:", audio_tensor.shape)   # should be (1, 13, 8, 32, 32)
                    #print("Encoder hidden state shape:", encoder_hidden_states.shape) # (1, 1024, 384)

                    # Run inference using ONNX model
                    output_latent = lip_syncer.run(None, {
                    'sample': audio_tensor.cpu().numpy(),
                    'timesteps': numpy.array([0], dtype=numpy.float16),
                    'encoder_hidden_states': encoder_hidden_states.astype(numpy.float16)
                })[0]

                    # Convert numpy array to torch tensor if needed
                    if isinstance(output_latent, numpy.ndarray):
                        output_latent = torch.from_numpy(output_latent).to(torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
					
                    # Convert Input: (1, 4, 8, 64, 64) to Output: (1, 3, 512, 512) for downstream transpose
                    close_vision_frame = normalize_latentsync_frame(output_latent)

                    # After model inference
                    print("🔍 Model output - min/max:", output_latent.min().item(), output_latent.max().item())
                    print("🔍 Model output - mean:", output_latent.mean().item())
                    print("🔍 Model output shape:", output_latent.shape)
            except Exception as e:
                logger.error(f"LatentSync processing failed: {str(e)}", __name__)
                return close_vision_frame
        else:
            # Wav2Lip-style direct inference with image and mel-spectrogram
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


# Prepare audio for LatentSync: log-mel normalization + reshape to (1, 13, 8, 64, 64)
# Input: (1, 1, 80, 16) mel spectrogram → Output: (1, 13, 8, 64, 64)
def prepare_latentsync_audio(temp_audio_frame: AudioFrame) -> torch.Tensor:
    """
    Converts mel spectrogram (1, 1, 80, 16) → (1, 13, 8, 64, 64), float16
    """
    try:
        #print("🔹 Step 1 — Raw temp_audio_frame shape:", temp_audio_frame.shape)
        frame = temp_audio_frame.squeeze()  # (80, 16)
        #print("🔹 Step 2 — Squeezed shape:", frame.shape)

        # Normalize like Whisper
        # Shifts and scales to roughly [0, 2]
        frame = numpy.maximum(frame, 1e-10)
        frame = numpy.log10(frame)
        frame = numpy.maximum(frame, frame.max() - 8.0)
        frame = (frame + 4.0) / 4.0
        frame = frame.astype(numpy.float32) # Normalise first in float32 for stability
        #print("Step 3 — Normalized. Shape:", frame.shape) # (80, 16)

        # Trim or pad time steps to 13
        time_dim = frame.shape[1]
        if time_dim < 13:
            pad = 13 - time_dim
            #print(f"ℹ️ Padding {pad} zeros")
            frame = numpy.pad(frame, ((0, 0), (0, pad)), mode='constant')
        elif time_dim > 13:
            #print(f"ℹ️ Trimming from {time_dim} to 13 time steps")
            frame = frame[:, :13]

        #print("🔹 Step 4 — Trimmed/padded shape:", frame.shape)  # (80, 13)

        blocks = []
        for t in range(13):
            slice_80 = frame[:, t]  # (80,)
            
			# Create a 3D cube (8, 64, 64) from the first 8 values
            cube = numpy.tile(slice_80[:8], (64, 64, 1)).transpose(2, 0, 1) # (8, 64, 64)
            blocks.append(cube)

        reshaped = numpy.stack(blocks, axis=0)  # (13, 8, 64, 64)
        #print("Step 5 — Created 3D blocks:", reshaped.shape)

        final = reshaped[numpy.newaxis, ...]  # (1, 13, 8, 64, 64)
        #print("✅ Final tensor shape:", final.shape)

        tensor = torch.from_numpy(final).to(torch.float16)
        return tensor.cuda() if torch.cuda.is_available() else tensor

    except Exception as e:
        raise RuntimeError(f"❌ Failed to prepare LatentSync audio: {str(e)}")


# Input: (H, W, 3) BGR image (OpenCV), Output: torch.Tensor (1, 4, 64, 64)
# Prepare video frame for LatentSync: resize, normalize, encode with VAE → latent shape (1, 4, 64, 64)
def prepare_latentsync_frame(vision_frame: VisionFrame) -> torch.Tensor:
    if vision_frame is None:
        raise ValueError("❌ vision_frame is None.")
    if not isinstance(vision_frame, numpy.ndarray):
        raise TypeError("❌ vision_frame is not a numpy array.")
    if vision_frame.size == 0:
        raise ValueError("❌ vision_frame is empty.")

    #print("🖼️ Raw vision_frame shape:", vision_frame.shape)

    try:
		# 🛡️ Ensure valid OpenCV input: uint8 in [0, 255]
        if vision_frame.dtype != numpy.uint8:
            print(f"⚠️ Warning: vision_frame dtype is {vision_frame.dtype}, converting to uint8.")
            vision_frame = numpy.clip(vision_frame, 0, 255).astype(numpy.uint8)

        # ✅ Convert from BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(vision_frame, cv2.COLOR_BGR2RGB)

        # After BGR to RGB conversion
        print("🔍 After BGR2RGB - min/max:", frame_rgb.min(), frame_rgb.max())
        print("🔍 After BGR2RGB - mean:", frame_rgb.mean())

        # ✅ Resize to 512x512
        resized = cv2.resize(frame_rgb, (512, 512))
        print("✅ Resized frame to 512x512")

        # ✅ Normalize to [-1, 1] in float32 then convert to float16
        normalized = resized.astype(numpy.float32) / 255.0
        normalized = (normalized * 2.0) - 1.0
        normalized = normalized.astype(numpy.float16)

        # After normalization
        print("🔍 After normalization - min/max:", normalized.min(), normalized.max())
        print("🔍 After normalization - mean:", normalized.mean())

        # ✅ Change shape to (1, 3, 512, 512)
        tensor = torch.from_numpy(numpy.transpose(normalized, (2, 0, 1))).unsqueeze(0)

        # ✅ Move to device
        tensor = tensor.to(torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ Encode with VAE
        with torch.no_grad():
            latent = vae.encode(tensor).latent_dist.sample().to(torch.float16) * 0.18215 # → (1, 4, 64, 64)
            latent = F.interpolate(latent, size = (64, 64), mode='bilinear', align_corners=False) # Ensure final output is (1, 4, 64, 64)
            print("🔍 VAE latent - min/max:", latent.min().item(), latent.max().item())
            print("🔍 VAE latent - mean:", latent.mean().item())

        return latent # this will be encoder_hidden_states (after reshape)

    except Exception as e:
        raise RuntimeError(f"❌ Failed to prepare vision frame: {str(e)}")


# Convert LatentSync UNet output latent back to displayable image (512x512x3 RGB)
# Input: (1, 4, 8, 64, 64) → Output: (1, 3, 512, 512) for downstream transpose
def normalize_latentsync_frame(latent: torch.Tensor) -> VisionFrame:
    if not isinstance(latent, torch.Tensor):
        raise TypeError("Input must be a torch tensor")
    
    try:
        if latent.ndim == 5:
            # ✅ Reduce temporal dim: (1, 4, 8, 64, 64) → (1, 4, 64, 64)
            latent = latent[:, :, 4, :, :]

        with torch.no_grad():
            print("🔍 Before VAE decode - latent min/max:", latent.min().item(), latent.max().item())
            print("🔍 Before VAE decode - latent mean:", latent.mean().item())

            latent = latent.to(torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")

            # Interpolate to 64x64 if needed just to be safe (Optional)
            latent = F.interpolate(latent, size=(64, 64), mode='bilinear', align_corners=False)
            latent = latent.clamp(-1, 1)
            
            # Decode from vae (already float16 compatible)
            decoded = vae.decode(latent / 0.18215).sample  # → (1, 3, 512, 512)

            print("🔍 After VAE decode - raw min/max:", decoded.min().item(), decoded.max().item())
            print("🔍 After VAE decode - raw mean:", decoded.mean().item())

            # Convert to uint8 image (0–255)
            decoded = (decoded.clamp(-1, 1) + 1) / 2.0
            decoded = decoded[0].permute(1, 2, 0).cpu().numpy() * 255  # → (512, 512, 3)
            decoded = decoded.astype(numpy.uint8)

            # 🔁 Convert RGB → BGR for OpenCV
            decoded = cv2.cvtColor(decoded, cv2.COLOR_RGB2BGR)

            print("🧪 Final normalized close frame - min/max per channel:", decoded[..., 0].min(), decoded[..., 1].min(), decoded[..., 2].min())
            print("🧪 Channel means:", decoded[..., 0].mean(), decoded[..., 1].mean(), decoded[..., 2].mean())

        return decoded

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

	for queue_payload in process_manager.manage(queue_payloads):
		frame_number = queue_payload.get('frame_number')
		target_vision_path = queue_payload.get('frame_path')
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
