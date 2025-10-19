import requests

print("Testing audio transcription API")

# Envoyer un fichier audio au serveur
audio_path = "test.wav"

with open(audio_path, "rb") as f:
    files = {
        "audio_file": (audio_path, f, "audio/wav")
    }  # Format attendu par requests (clé, nom_fichier, contenu fichier, type_mime)
    response = requests.post("http://127.0.0.1:8000/transcribe", files=files)

if response.status_code == 200:
    result = response.json()
    print(f"Fichier: {result['filename']}")
    print(f"Langue: {result['language']}")
    print(f"Texte: {result['text']}")
else:
    print(f"Erreur: {response.json()}")
