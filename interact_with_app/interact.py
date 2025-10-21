import requests
from tkinter import Tk, filedialog

print("Testing audio transcription API")

# URL de votre API déployée
API_URL = "https://audio-to-text-user-elliot.lab.sspcloud.fr/transcribe"
# API_URL = "http://localhost:8000/transcribe"

# Ouvrir l'explorateur de fichiers pour choisir un fichier audio
root = Tk()
root.withdraw()  # Cacher la fenêtre principale de tkinter
root.attributes("-topmost", True)  # Mettre la fenêtre au premier plan

print("Veuillez sélectionner un fichier audio...")
audio_path = filedialog.askopenfilename(
    title="Sélectionnez un fichier audio",
    filetypes=[
        ("Fichiers audio", "*.wav *.mp3 *.m4a *.ogg"),
        ("Fichiers WAV", "*.wav"),
        ("Fichiers MP3", "*.mp3"),
        ("Tous les fichiers", "*.*"),
    ],
)

# Vérifier si un fichier a été sélectionné
if not audio_path:
    print("Aucun fichier sélectionné. Arrêt du programme.")
    exit()

print(f"\n Fichier sélectionné: {audio_path}")
print(f"Envoi à l'API...")

try:
    with open(audio_path, "rb") as f:
        files = {"audio_file": (audio_path, f, "audio/wav")}
        response = requests.post(API_URL, files=files, timeout=120)

    if response.status_code == 200:
        result = response.json()
        print(f"\n Transcription réussie!")
        print(f"Fichier: {result['filename']}")
        print(f"Langue: {result['language']}")
        print(f"Texte: {result['text']}")
    else:
        print(f"Erreur {response.status_code}: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"Erreur de connexion: {e}")
except Exception as e:
    print(f"Erreur: {e}")
