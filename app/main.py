from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import whisper
import os
import tempfile
import asyncio

# ...existing code...
app = FastAPI()

DEVICE = "cpu"
MODEL_SIZE = "tiny"

print(f"Chargement du modèle Whisper {MODEL_SIZE} sur {DEVICE}...")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)
print("Modèle chargé avec succès!")

# Nouvelle interface HTML pour la racine
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Transcription Audio - Whisper</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }
        .card { border: 1px solid #eee; padding: 20px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }
        h1 { margin-top: 0; }
        .row { display:flex; gap:10px; align-items:center; }
        #result { white-space: pre-wrap; margin-top: 16px; background:#f8f8f8; padding:12px; border-radius:6px; }
        button { padding: 8px 14px; cursor:pointer; }
        input[type=file] { display:inline-block; }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>Transcription Audio avec Whisper</h1>
        <p>Sélectionnez un fichier audio (.wav ou .mp3) puis cliquez sur "Transcrire".</p>
        <div class="row">
          <input id="audioInput" type="file" accept=".wav,.mp3" />
          <button id="uploadBtn">Transcrire</button>
        </div>
        <div id="status"></div>
        <h3>Résultat</h3>
        <div id="result">Aucune transcription pour le moment.</div>
      </div>

      <script>
        const input = document.getElementById('audioInput');
        const btn = document.getElementById('uploadBtn');
        const status = document.getElementById('status');
        const result = document.getElementById('result');

        btn.addEventListener('click', async () => {
          const file = input.files[0];
          if (!file) {
            alert('Veuillez sélectionner un fichier audio.');
            return;
          }
          status.textContent = 'Envoi en cours...';
          result.textContent = '';
          const form = new FormData();
          // le nom du champ doit correspondre au paramètre python: audio_file
          form.append('audio_file', file, file.name);

          try {
            const resp = await fetch('/transcribe', {
              method: 'POST',
              body: form
            });
            if (!resp.ok) {
              const err = await resp.json().catch(()=>({detail: resp.statusText}));
              status.textContent = 'Erreur: ' + (err.detail || resp.statusText);
              return;
            }
            const data = await resp.json();
            status.textContent = 'Transcription terminée.';
            result.textContent = `Fichier: ${data.filename}\nLangue: ${data.language}\n\nTexte:\n${data.text}`;
          } catch (e) {
            status.textContent = 'Erreur réseau: ' + e.message;
          }
        });
      </script>
    </body>
    </html>
    """

# Endpoint pour la transcription audio (corrigé pour sauvegarder le fichier uploadé)
@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    allowed_extensions = [".wav", ".mp3"]
    file_extension = os.path.splitext(audio_file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté. Formats acceptés: {', '.join(allowed_extensions)}",
        )

    try:
        # sauvegarder dans un fichier temporaire
        suffix = file_extension if file_extension else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await audio_file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # transcrire depuis le fichier temporaire
        print(f"Transcription de {audio_file.filename} (temp: {tmp_path})...")
        # whisper attend un chemin de fichier ou un tableau audio
        result = model.transcribe(tmp_path)

        # supprimer le fichier temporaire
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return JSONResponse(
            content={
                "filename": audio_file.filename,
                "text": result.get("text", ""),
                "language": result.get("language", "unknown"),
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la transcription: {str(e)}")
