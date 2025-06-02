from functools import lru_cache
from typing import Any, List, Optional

import numpy
import scipy
from numpy._typing import NDArray
import soundfile as sf
import librosa  # For audio resampling

from facefusion.ffmpeg import read_audio_buffer
from facefusion.filesystem import is_audio
from facefusion.typing import Audio, AudioFrame, Fps, Mel, MelFilterBank, Spectrogram
from facefusion.voice_extractor import batch_extract_voice


@lru_cache(maxsize = 128)
def read_static_audio(audio_path : str, fps : Fps) -> Optional[List[AudioFrame]]:
	return read_audio(audio_path, fps)


def read_audio(audio_path : str, fps : Fps) -> Optional[List[AudioFrame]]:
	sample_rate = 48000
	channel_total = 2

	if is_audio(audio_path):
		audio_buffer = read_audio_buffer(audio_path, sample_rate, channel_total)
		audio = numpy.frombuffer(audio_buffer, dtype = numpy.int16).reshape(-1, 2)
		audio = prepare_audio(audio)
		spectrogram = create_spectrogram(audio)
		audio_frames = extract_audio_frames(spectrogram, fps)
		return audio_frames
	return None


@lru_cache(maxsize = 128)
def read_static_voice(audio_path : str, fps : Fps) -> Optional[List[AudioFrame]]:
	return read_voice(audio_path, fps)


def read_voice(audio_path : str, fps : Fps) -> Optional[List[AudioFrame]]:
	sample_rate = 48000
	channel_total = 2
	chunk_size = 240 * 1024
	step_size = 180 * 1024

	if is_audio(audio_path):
		audio_buffer = read_audio_buffer(audio_path, sample_rate, channel_total)
		audio = numpy.frombuffer(audio_buffer, dtype = numpy.int16).reshape(-1, 2)
		audio = batch_extract_voice(audio, chunk_size, step_size)
		audio = prepare_voice(audio)
		spectrogram = create_spectrogram(audio)
		audio_frames = extract_audio_frames(spectrogram, fps)
		return audio_frames
	return None


def get_audio_frame(audio_path : str, fps : Fps, frame_number : int = 0) -> Optional[AudioFrame]:
	if is_audio(audio_path):
		audio_frames = read_static_audio(audio_path, fps)
		if frame_number in range(len(audio_frames)):
			return audio_frames[frame_number]
	return None


def get_voice_frame(audio_path : str, fps : Fps, frame_number : int = 0) -> Optional[AudioFrame]:
	if is_audio(audio_path):
		voice_frames = read_static_voice(audio_path, fps)
		if frame_number in range(len(voice_frames)):
			return voice_frames[frame_number]
	return None


def create_empty_audio_frame() -> AudioFrame:
	mel_filter_total = 80
	step_size = 16
	audio_frame = numpy.zeros((mel_filter_total, step_size)).astype(numpy.int16)
	return audio_frame


def get_raw_audio_frame(audio_path : str, fps : Fps, frame_number : int) -> Optional[numpy.ndarray]:
	"""
	Get raw audio frame for a specific frame number (for LatentSync)
	🔧 CRITICAL FIX: Ensure minimum audio length to prevent Whisper padding errors
	"""
	if is_audio(audio_path):
		# LatentSync uses 16kHz for Whisper
		sample_rate = 16000
		frame_duration = 1.0 / fps
		frame_duration_samples = int(sample_rate * frame_duration)
		
		# 🔧 CRITICAL FIX: Whisper requires minimum 400ms of audio (6400 samples at 16kHz)
		min_samples = 6400  # 400ms at 16kHz - Whisper's minimum requirement
		# 🔧 CRITICAL FIX: For very short frames, use larger window for Whisper stability
		audio_window_samples = max(min_samples, frame_duration_samples * 4)  # Use 4x frame duration minimum
		
		# Use larger window for Whisper compatibility
		start_sample = max(0, int(frame_number * frame_duration_samples) - audio_window_samples // 2)
		
		try:
			# Read the audio file
			audio_data, original_sample_rate = sf.read(audio_path)
			
			# Ensure mono
			if len(audio_data.shape) > 1:
				audio_data = numpy.mean(audio_data, axis=1)
			
			# Resample if needed
			if original_sample_rate != sample_rate:
				audio_data = resample_audio(audio_data, original_sample_rate, sample_rate)
			
			# Extract the audio window
			end_sample = start_sample + audio_window_samples
			
			if start_sample < len(audio_data):
				audio_frame = audio_data[start_sample:end_sample]
				
				# 🔧 CRITICAL FIX: Ensure minimum length by padding or repeating
				if len(audio_frame) < min_samples:
					print(f"🔧 Audio frame too short ({len(audio_frame)} samples), extending to {min_samples}")
					if len(audio_frame) > 0:
						# Repeat the audio to reach minimum length
						repeat_count = (min_samples // len(audio_frame)) + 1
						audio_frame = numpy.tile(audio_frame, repeat_count)[:min_samples]
					else:
						# Create minimal noise if empty
						audio_frame = numpy.random.normal(0, 0.001, min_samples).astype(numpy.float32)
				
				print(f"🔧 Raw audio frame: {len(audio_frame)} samples for frame {frame_number}")
				return audio_frame.astype(numpy.float32)
			else:
				# 🔧 CRITICAL FIX: Create minimum-size audio frame if beyond audio length
				print(f"🔧 Frame {frame_number} beyond audio length, creating minimum audio frame")
				return numpy.random.normal(0, 0.001, min_samples).astype(numpy.float32)
				
		except Exception as e:
			print(f"❌ Error reading audio frame: {e}")
			# 🔧 CRITICAL FIX: Return minimum-size frame on error
			return numpy.random.normal(0, 0.001, min_samples).astype(numpy.float32)
	
	return None


def create_empty_raw_audio_frame(fps: Fps, sample_rate: int = 16000) -> numpy.ndarray:
	"""Create empty raw audio frame for LatentSync (FP32 format)"""
	frame_duration_samples = int(sample_rate / fps)
	return numpy.zeros(frame_duration_samples, dtype=numpy.float32)


def prepare_audio(audio : Audio) -> Audio:
	if audio.ndim > 1:
		audio = numpy.mean(audio, axis = 1)
	audio = audio / numpy.max(numpy.abs(audio), axis = 0)
	audio = scipy.signal.lfilter([ 1.0, -0.97 ], [ 1.0 ], audio)
	return audio


def prepare_voice(audio : Audio) -> Audio:
	sample_rate = 48000
	resample_rate = 16000

	audio = scipy.signal.resample(audio, int(len(audio) * resample_rate / sample_rate))
	audio = prepare_audio(audio)
	return audio


def convert_hertz_to_mel(hertz : float) -> float:
	return 2595 * numpy.log10(1 + hertz / 700)


def convert_mel_to_hertz(mel : Mel) -> NDArray[Any]:
	return 700 * (10 ** (mel / 2595) - 1)


def create_mel_filter_bank() -> MelFilterBank:
	mel_filter_total = 80
	mel_bin_total = 800
	sample_rate = 16000
	min_frequency = 55.0
	max_frequency = 7600.0
	mel_filter_bank = numpy.zeros((mel_filter_total, mel_bin_total // 2 + 1))
	mel_frequency_range = numpy.linspace(convert_hertz_to_mel(min_frequency), convert_hertz_to_mel(max_frequency), mel_filter_total + 2)
	indices = numpy.floor((mel_bin_total + 1) * convert_mel_to_hertz(mel_frequency_range) / sample_rate).astype(numpy.int16)

	for index in range(mel_filter_total):
		start = indices[index]
		end = indices[index + 1]
		mel_filter_bank[index, start:end] = scipy.signal.windows.triang(end - start)
	return mel_filter_bank


def create_spectrogram(audio : Audio) -> Spectrogram:
	mel_bin_total = 800
	mel_bin_overlap = 600
	mel_filter_bank = create_mel_filter_bank()
	spectrogram = scipy.signal.stft(audio, nperseg = mel_bin_total, nfft = mel_bin_total, noverlap = mel_bin_overlap)[2]
	spectrogram = numpy.dot(mel_filter_bank, numpy.abs(spectrogram))
	return spectrogram


def extract_audio_frames(spectrogram : Spectrogram, fps : Fps) -> List[AudioFrame]:
	mel_filter_total = 80
	step_size = 16
	audio_frames = []
	indices = numpy.arange(0, spectrogram.shape[1], mel_filter_total / fps).astype(numpy.int16)
	indices = indices[indices >= step_size]

	for index in indices:
		start = max(0, index - step_size)
		audio_frames.append(spectrogram[:, start:index])
	return audio_frames


@lru_cache(maxsize = 128)
def read_static_raw_audio_for_latentsync(audio_path: str, fps: Fps) -> Optional[numpy.ndarray]:
	"""Read entire raw audio file for LatentSync batch processing (cached)"""
	return read_raw_audio_for_latentsync(audio_path, fps)


def read_raw_audio_for_latentsync(audio_path: str, fps: Fps) -> Optional[numpy.ndarray]:
	"""Read entire raw audio file for LatentSync batch processing"""
	if is_audio(audio_path):
		# LatentSync uses 16kHz for Whisper
		sample_rate = 16000
		channel_total = 2
		
		audio_buffer = read_audio_buffer(audio_path, sample_rate, channel_total)
		audio = numpy.frombuffer(audio_buffer, dtype = numpy.int16).reshape(-1, 2)
		audio = prepare_audio(audio)  # Convert to mono and normalize
		
		# Ensure FP32 consistency for LatentSync
		audio = audio.astype(numpy.float32)
		
		return audio
	return None


def get_audio_chunks_for_latentsync(audio_path: str, fps: Fps, total_frames: int) -> List[numpy.ndarray]:
	"""
	Get audio chunks for LatentSync batch processing (official approach)
	This mimics the official audio2feat + feature2chunks workflow
	🔧 CRITICAL FIX: Ensure minimum audio length to prevent Whisper padding errors
	"""
	if is_audio(audio_path):
		# Read the entire audio file once (like official audio2feat)
		full_audio = read_static_raw_audio_for_latentsync(audio_path, fps)
		if full_audio is None:
			return []
		
		# Calculate frame duration in samples
		sample_rate = 16000  # LatentSync uses 16kHz
		frame_duration_samples = int(sample_rate / fps)
		
		# 🔧 CRITICAL FIX: Whisper requires minimum 400ms of audio (6400 samples at 16kHz)
		min_samples = 6400  # 400ms at 16kHz - Whisper's minimum requirement
		# 🔧 CRITICAL FIX: Use larger window for better Whisper stability
		audio_window_samples = max(min_samples, frame_duration_samples * 8)  # Use 8x frame duration for stability
		
		print(f"🔧 Audio chunking: {total_frames} frames, {audio_window_samples} samples per chunk")
		
		# Create chunks for each frame (like official feature2chunks)
		audio_chunks = []
		for frame_idx in range(total_frames):
			# Calculate start position (center the window around the target frame)
			target_center = frame_idx * frame_duration_samples + frame_duration_samples // 2
			start_sample = max(0, target_center - audio_window_samples // 2)
			end_sample = start_sample + audio_window_samples
			
			# Extract the audio segment
			if start_sample < len(full_audio):
				audio_chunk = full_audio[start_sample:end_sample]
				
				# 🔧 CRITICAL FIX: Ensure minimum length by repeating if necessary
				if len(audio_chunk) < min_samples:
					print(f"🔧 Chunk {frame_idx} too short ({len(audio_chunk)} samples), extending to {min_samples}")
					if len(audio_chunk) > 0:
						# Repeat the audio to reach minimum length
						repeat_count = (min_samples // len(audio_chunk)) + 1
						audio_chunk = numpy.tile(audio_chunk, repeat_count)[:min_samples]
					else:
						# Create minimal noise if empty
						audio_chunk = numpy.random.normal(0, 0.001, min_samples).astype(numpy.float32)
				
				# Pad with zeros if still needed (at the end)
				if len(audio_chunk) < audio_window_samples:
					padding_needed = audio_window_samples - len(audio_chunk)
					audio_chunk = numpy.pad(audio_chunk, (0, padding_needed), mode='constant', constant_values=0)
				
				audio_chunks.append(audio_chunk.astype(numpy.float32))
			else:
				# 🔧 CRITICAL FIX: Create minimum-size chunk if beyond audio length
				print(f"🔧 Frame {frame_idx} beyond audio, creating minimum chunk")
				audio_chunk = numpy.random.normal(0, 0.001, audio_window_samples).astype(numpy.float32)
				audio_chunks.append(audio_chunk)
		
		print(f"✅ Created {len(audio_chunks)} audio chunks, each {audio_window_samples} samples")
		return audio_chunks
	return []


def get_audio_chunk_from_batch(audio_chunks: List[numpy.ndarray], frame_number: int) -> Optional[numpy.ndarray]:
	"""Get a specific audio chunk from a batch of pre-computed chunks"""
	if audio_chunks and frame_number < len(audio_chunks):
		return audio_chunks[frame_number]
	return None


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
			import scipy.signal
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
