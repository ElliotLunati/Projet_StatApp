from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import os
import torch

# Crée une instance FastAPI
app = FastAPI()

# Vérifier la disponibilité du GPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"GPU détecté: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    DEVICE = "cpu"
    print("Aucun GPU détecté, utilisation du CPU")

# Charger le modèle Whisper au démarrage du serveur
MODEL_SIZE = "tiny"  # tiny, base, small, medium, large, turbo

print(f"Chargement du modèle Whisper {MODEL_SIZE} sur {DEVICE}...")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)
print("Modèle chargé avec succès!")


@app.get("/")
def read_root():
    return {
        "message": "Serveur de transcription audio avec Whisper",
        "Model": MODEL_SIZE,
        "Device": DEVICE,
    }


# Endpoint pour la transcription audio
@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
):  # File(...) => fichier obligatoire
    """
    Reçoit un fichier audio et retourne la transcription texte
    """
    # Attributs UploadFile de audio_file
    # audio_file.filename       str - "test.wav"
    # audio_file.content_type   str - "audio/wav"
    # audio_file.file          SpooledTemporaryFile - objet fichier
    # audio_file.size          int - taille en bytes

    # Méthodes asynchrones UploadFile de audio_file
    # await audio_file.read()        bytes - lit tout le contenu
    # await audio_file.write(data)   écrit des données
    # await audio_file.seek(0)       repositionne au début
    # await audio_file.close()       ferme le fichier

    # Vérifier le type de fichier
    allowed_extensions = [".wav", ".mp3"]
    file_extension = os.path.splitext(audio_file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté. Formats acceptés: {', '.join(allowed_extensions)}",
        )

    # Transcription de l'audio uploadé avec whisper
    try:
        print(f"Transcription de {audio_file.filename}...")
        result = model.transcribe(audio_file.filename)

        return JSONResponse(
            content={
                "filename": audio_file.filename,
                "text": result["text"],
                "language": result.get("language", "unknown"),
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la transcription: {str(e)}"
        )
