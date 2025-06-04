from typing import List, Optional, Tuple

import gradio

from facefusion import state_manager, wording
from facefusion.common_helper import get_first
from facefusion.filesystem import filter_audio_paths, filter_image_paths, has_audio, has_image
from facefusion.uis.core import register_ui_component
from facefusion.uis.typing import File

SOURCE_FILE : Optional[gradio.File] = None
SOURCE_AUDIO : Optional[gradio.Audio] = None
SOURCE_IMAGE : Optional[gradio.Image] = None


def render() -> None:
	global SOURCE_FILE
	global SOURCE_AUDIO
	global SOURCE_IMAGE

	print(f"🔍 DEBUG: source.py render() called")
	current_source_paths = state_manager.get_item('source_paths')
	print(f"   - Initial source_paths: {current_source_paths}")
	
	has_source_audio = has_audio(state_manager.get_item('source_paths'))
	has_source_image = has_image(state_manager.get_item('source_paths'))
	print(f"   - has_source_audio: {has_source_audio}")
	print(f"   - has_source_image: {has_source_image}")
	
	SOURCE_FILE = gradio.File(
		label = wording.get('uis.source_file'),
		file_count = 'multiple',
		file_types =
		[
			'audio',
			'image'
		],
		value = state_manager.get_item('source_paths') if has_source_audio or has_source_image else None
	)
	print(f"   - SOURCE_FILE created with value: {SOURCE_FILE.value}")
	
	source_file_names = [ source_file_value.get('path') for source_file_value in SOURCE_FILE.value ] if SOURCE_FILE.value else None
	source_audio_path = get_first(filter_audio_paths(source_file_names))
	source_image_path = get_first(filter_image_paths(source_file_names))
	print(f"   - source_audio_path: {source_audio_path}")
	print(f"   - source_image_path: {source_image_path}")
	
	SOURCE_AUDIO = gradio.Audio(
		value = source_audio_path if has_source_audio else None,
		visible = has_source_audio,
		show_label = False
	)
	SOURCE_IMAGE = gradio.Image(
		value = source_image_path if has_source_image else None,
		visible = has_source_image,
		show_label = False
	)
	register_ui_component('source_audio', SOURCE_AUDIO)
	register_ui_component('source_image', SOURCE_IMAGE)


def listen() -> None:
	print(f"🔍 DEBUG: source.py listen() called - registering SOURCE_FILE change handler")
	SOURCE_FILE.change(update, inputs = SOURCE_FILE, outputs = [ SOURCE_AUDIO, SOURCE_IMAGE ])


def update(files : List[File]) -> Tuple[gradio.Audio, gradio.Image]:
	print(f"🔍 DEBUG source.py update called:")
	print(f"   - files: {files}")
	print(f"   - files type: {type(files)}")
	
	file_names = [ file.name for file in files ] if files else None
	print(f"   - file_names: {file_names}")
	
	# 🔧 CRITICAL FIX: Copy Gradio temp files to persistent location
	if file_names:
		import shutil
		import tempfile
		import os
		import subprocess
		
		persistent_file_names = []
		for file_name in file_names:
			if file_name and os.path.exists(file_name):
				file_ext = os.path.splitext(file_name)[1].lower()
				
				# 🔧 AUDIO CONVERSION FIX: Convert audio files to clean WAV format
				if file_ext in ['.mp3', '.m4a', '.aac', '.ogg', '.flac', '.wav']:
					print(f"🔧 Audio file detected: {file_name} ({file_ext})")
					
					# Create persistent WAV file
					persistent_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
					persistent_path = persistent_file.name
					persistent_file.close()
					
					# Convert to 16kHz mono WAV using FFmpeg
					try:
						ffmpeg_cmd = [
							'ffmpeg', '-y', '-i', file_name,
							'-ar', '16000',  # 16kHz sample rate
							'-ac', '1',      # Mono
							'-f', 'wav',     # WAV format
							'-acodec', 'pcm_s16le',  # 16-bit PCM
							persistent_path
						]
						
						print(f"🔧 Converting audio: {file_ext} → WAV (16kHz mono)")
						result = subprocess.run(ffmpeg_cmd, capture_output=True, capture_stderr=True, text=True)
						
						if result.returncode == 0:
							persistent_file_names.append(persistent_path)
							print(f"🔧 Audio converted: {file_name} → {persistent_path}")
							
							# Verify the converted file
							if os.path.exists(persistent_path) and os.path.getsize(persistent_path) > 0:
								print(f"✅ Converted WAV file verified: {os.path.getsize(persistent_path)} bytes")
							else:
								print(f"❌ Converted WAV file verification failed")
								persistent_file_names.pop()  # Remove from list
								persistent_file_names.append(file_name)  # Fallback to original
						else:
							print(f"❌ FFmpeg conversion failed: {result.stderr}")
							print(f"🔧 Falling back to direct copy")
							os.unlink(persistent_path)  # Remove failed WAV file
							
							# Fallback to direct copy
							file_ext = os.path.splitext(file_name)[1] or '.tmp'
							persistent_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
							persistent_path = persistent_file.name
							persistent_file.close()
							shutil.copy2(file_name, persistent_path)
							persistent_file_names.append(persistent_path)
							print(f"🔧 Direct copy fallback: {file_name} → {persistent_path}")
							
					except Exception as e:
						print(f"❌ Audio conversion error: {e}")
						print(f"🔧 Falling back to direct copy")
						
						# Fallback to direct copy
						file_ext = os.path.splitext(file_name)[1] or '.tmp'
						persistent_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
						persistent_path = persistent_file.name
						persistent_file.close()
						shutil.copy2(file_name, persistent_path)
						persistent_file_names.append(persistent_path)
						print(f"🔧 Direct copy fallback: {file_name} → {persistent_path}")
				else:
					# For non-audio files (images), use direct copy
					file_ext = os.path.splitext(file_name)[1] or '.tmp'
					persistent_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
					persistent_path = persistent_file.name
					persistent_file.close()
					
					# Copy the file to persistent location
					try:
						shutil.copy2(file_name, persistent_path)
						persistent_file_names.append(persistent_path)
						print(f"🔧 Copied: {file_name} → {persistent_path}")
					except Exception as e:
						print(f"❌ Failed to copy {file_name}: {e}")
						persistent_file_names.append(file_name)  # Fallback to original
			else:
				print(f"⚠️ File does not exist: {file_name}")
				persistent_file_names.append(file_name)  # Keep original path
		
		file_names = persistent_file_names
		print(f"   - persistent file_names: {file_names}")
	
	has_source_audio = has_audio(file_names)
	has_source_image = has_image(file_names)
	print(f"   - has_source_audio: {has_source_audio}")
	print(f"   - has_source_image: {has_source_image}")
	
	if has_source_audio or has_source_image:
		source_audio_path = get_first(filter_audio_paths(file_names))
		source_image_path = get_first(filter_image_paths(file_names))
		print(f"   - source_audio_path: {source_audio_path}")
		print(f"   - source_image_path: {source_image_path}")
		
		print(f"🔧 Setting source_paths to: {file_names}")
		state_manager.set_item('source_paths', file_names)
		
		# Verify it was set
		current_source_paths = state_manager.get_item('source_paths')
		print(f"✅ Verified source_paths set to: {current_source_paths}")
		
		return gradio.Audio(value = source_audio_path, visible = has_source_audio), gradio.Image(value = source_image_path, visible = has_source_image)
	
	print(f"⚠️ No valid audio or image files found, clearing source_paths")
	state_manager.clear_item('source_paths')
	return gradio.Audio(value = None, visible = False), gradio.Image(value = None, visible = False)
