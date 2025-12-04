import soundfile as sf
import numpy as np
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# ===== CONFIG =====
MODEL = "omniASR_LLM_3B"  # any model
DEVICE = "cuda"
CHUNK_SEC = 30            # must be < 40 seconds
OVERLAP_SEC = 1.0         # to avoid cutting words
LANG = "arz_Arab"               # or "eng_Latn"
# ===================


def chunk_audio(waveform, sr, chunk_sec, overlap_sec):
    chunk_size = int(chunk_sec * sr)
    overlap = int(overlap_sec * sr)
    step = chunk_size - overlap

    chunks = []
    for start in range(0, len(waveform), step):
        end = min(start + chunk_size, len(waveform))
        chunks.append(waveform[start:end])
        if end == len(waveform):
            break
    return chunks


def transcribe_long_audio(path):
    # Load audio file
    audio, sr = sf.read(path)
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    print(f"Loaded {path}, duration: {len(audio)/sr:.2f}s")

    # Chunk audio
    chunks = chunk_audio(audio, sr, CHUNK_SEC, OVERLAP_SEC)

    pipeline = ASRInferencePipeline(model_card=MODEL, device=DEVICE)

    final_text = ""
    last_tail = ""

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}...")
        audio_input = [{"waveform": chunk, "sample_rate": sr}]

        out = pipeline.transcribe(
            audio_input,
            lang=[LANG] if LANG else None,
            batch_size=1
        )[0]

        # Simple stitching: avoid duplicating overlap words
        if last_tail and out.startswith(last_tail):
            out = out[len(last_tail):]

        final_text += " " + out.strip()

        # Store last few words of this chunk as tail
        last_tail = " ".join(out.split()[-10:])
        print(f"Final text: {final_text}")

    return final_text.strip()


if __name__ == "__main__":
    AUDIO_FILE = "../audios/medconv.wav"
    text = transcribe_long_audio(AUDIO_FILE)
    print("\n==== FINAL TRANSCRIPT ====\n")
    print(text)
