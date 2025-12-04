import os
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# --- Configuration ---

# ⚠️ IMPORTANT: Replace this with the actual path to your Egyptian Arabic audio file.
# The audio must be an uncompressed format like WAV or FLAC.
# Note: Currently, the model is limited to audio files shorter than 40 seconds for inference.
AUDIO_FILE_PATH = "../audios/medconv.wav"

# The Model Card specifies the Omnilingual ASR model to use. 
# 'omniASR_LLM_7B' is the largest and most accurate model.
MODEL_CARD = "omniASR_LLM_3B"

# Language code for Arabic (Standard Arabic, or a general Arabic representation) 
# The Omnilingual model is designed to generalize to dialects. 
# You will need to check the official documentation for the *specific* code 
# representing Egyptian Arabic if one is available, otherwise use a general Arabic code.
# 'arb_Arab' or a dialect-specific code like 'arz_Arab' for Egyptian Arabic (if supported).
# For this example, we'll use a likely candidate for Arabic script:
LANGUAGE_CODE = "arz_Arab" 

# --- ASR Transcription ---

# 1. Initialize the ASR Inference Pipeline
# The model will be automatically downloaded on first run (around 35-40 GB for the 7B model).
print(f"Loading ASR pipeline with model: {MODEL_CARD}...")
pipeline = ASRInferencePipeline(model_card=MODEL_CARD)
print("Pipeline loaded successfully.")

# 2. Prepare the input data
audio_files = [AUDIO_FILE_PATH]
# Pass a list of language codes corresponding to the audio files
langs = [LANGUAGE_CODE]

# 3. Transcribe the audio
print(f"Starting transcription for {os.path.basename(AUDIO_FILE_PATH)} in language {LANGUAGE_CODE}...")
transcriptions = pipeline.transcribe(
    audio_files, 
    lang=langs, 
    # Batch size is 1 since we are only processing one file
    batch_size=1 
)

# 4. Display the result
if transcriptions:
    transcribed_text = transcriptions[0]
    print("\n--- Transcription Result ---")
    print(f"Language: {LANGUAGE_CODE}")
    print(f"Predicted Text: {transcribed_text}")
    print("--------------------------")
else:
    print("Transcription failed or returned no results.")