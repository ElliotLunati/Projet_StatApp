import whisper

AUDIO_PATH = "test.wav"
MODEL_SIZE = "turbo"  # tiny, base, small, medium, large, turbo
DEVICE = "cuda"  # Mettre "cpu" si pas de GPU dispo

model = whisper.load_model(MODEL_SIZE, device=DEVICE)
result = model.transcribe(AUDIO_PATH)
print(result["text"])
