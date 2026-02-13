import warnings

from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import whisper
import os
import tempfile
import torch
import torchaudio
from pyannote.audio import Pipeline


# Crée une instance FastAPI
app = FastAPI()

# Montage du dossier images pour servir les fichiers statiques
app.mount(
    "/images",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "images")),
    name="images",
)

# Vérification de la disponibilité du GPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    print(f"Nombre de GPUs: {torch.cuda.device_count()}")
else:
    DEVICE = torch.device("cpu")
    print("GPU non disponible, utilisation du CPU")

# Charger le modèle Whisper au démarrage du serveur
MODEL_SIZE = "tiny"  # tiny, base, small, medium, large, turbo

print(f"Chargement du modèle Whisper {MODEL_SIZE} sur {DEVICE}")
model = whisper.load_model(MODEL_SIZE, device=str(DEVICE))
print("Modèle Whisper chargé avec succès!")

# Charger la pipeline de diarisation au démarrage du serveur
print("Chargement de la pipeline de diarisation")

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    diarization_model_path = os.path.join(
        current_dir, "pyannote-speaker-diarization-community-1"
    )
    diarization_pipeline = Pipeline.from_pretrained(diarization_model_path)
    diarization_pipeline.to(DEVICE)
    print("Pipeline de diarisation chargée avec succès")

except Exception as e:
    print("\n Erreur : les modèles de diarisation n'ont pas pu être chargés.")
    print(e)

# GET : get info / return info (lecture de données uniquement)
# POST : créer ressource dans db (envoie de données)
# PUT : mettre à jour ressource dans db
# DELETE : supprimer ressource dans db


@app.get("/")
def read_root():
    """Interface web pour uploader et transcrire des fichiers audio"""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    return FileResponse(template_path)


# Endpoint pour la transcription audio
@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
):  # File(...) => fichier obligatoire
    """
    Reçoit un fichier audio et retourne la transcription texte
    """
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

        # Créer un fichier temporaire pour sauvegarder l'audio uploadé
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            # Lire et écrire le contenu du fichier uploadé
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Transcrire avec le chemin du fichier temporaire
            result = model.transcribe(temp_path)

            # Liste dee la forme :
            # [{"id": 0,"start": 0.0,"end": 3.5,"text": " Hello, how are you?"}, ..]
            segments = result.get("segments", [])

            for seg in segments:
                print(
                    f"[{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text'].strip()}"
                )

            # Diarisation des locuteurs
            diarization_result = []
            if diarization_pipeline:
                try:
                    print("Diarisation en cours...")
                    # Charger l'audio avec torchaudio
                    waveform, sample_rate = torchaudio.load(temp_path)

                    # Préparer l'entrée au format dictionnaire
                    audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

                    # Lancer la diarisation
                    output = diarization_pipeline(audio_in_memory)

                    # Formater les résultats
                    for turn, speaker in output.speaker_diarization:
                        # Trouver le texte correspondant à ce segment temporel
                        segment_text = ""
                        for seg in segments:
                            # Vérifier si le segment Whisper chevauche le segment de diarisation
                            if seg["start"] < turn.end and seg["end"] > turn.start:
                                segment_text += seg["text"]

                        diarization_result.append(
                            {
                                "start": turn.start,
                                "end": turn.end,
                                "speaker": speaker,
                                "text": segment_text.strip(),
                            }
                        )
                    print(
                        f"Diarisation terminée: {len(diarization_result)} segments trouvés"
                    )
                    for segment in diarization_result:
                        print(
                            f"start={segment['start']:.1f}s stop={segment['end']:.1f}s {segment['speaker']} text='{segment['text']}'"
                        )

                except Exception as e:
                    print(f"Erreur lors de la diarisation: {e}")

            return JSONResponse(
                content={
                    "filename": audio_file.filename,
                    "text": result["text"],
                    "language": result.get("language", "unknown"),
                    "diarization": diarization_result if diarization_result else None,
                }
            )
        finally:
            # Nettoyer le fichier temporaire
            os.unlink(temp_path)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la transcription: {str(e)}"
        )
