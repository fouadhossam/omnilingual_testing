from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

arab_langs = [lang for lang in supported_langs if lang.endswith("_Arab")]

print(f"Total _Arab languages: {len(arab_langs)}")
print(arab_langs)
