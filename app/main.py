from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import whisper
import os
import tempfile

# GET : get info / return info (lecture de données uniquement)
# POST : créer ressource dans db (envoie de données)
# PUT : mettre à jour ressource dans db
# DELETE : supprimer ressource dans db

# Crée une instance FastAPI
app = FastAPI()

# On utilise le GPU disponible avec ssp cloud
DEVICE = "cuda"

# Charger le modèle Whisper au démarrage du serveur
MODEL_SIZE = "tiny"  # tiny, base, small, medium, large, turbo

print(f"Chargement du modèle Whisper {MODEL_SIZE} sur {DEVICE}...")
model = whisper.load_model(MODEL_SIZE, device=DEVICE)
print("Modèle chargé avec succès!")


@app.get("/", response_class=HTMLResponse)
def read_root():
    """Interface web pour uploader et transcrire des fichiers audio"""
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Transcription Audio - Whisper</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }

            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
                max-width: 800px;
                width: 100%;
            }

            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 2em;
            }

            .subtitle {
                color: #666;
                margin-bottom: 30px;
            }

            .drop-zone {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 60px 20px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
                background: #f8f9ff;
            }

            .drop-zone:hover, .drop-zone.dragover {
                background: #e8ebff;
                border-color: #764ba2;
                transform: scale(1.02);
            }

            .drop-zone-icon {
                font-size: 4em;
                margin-bottom: 20px;
            }

            .drop-zone-text {
                color: #667eea;
                font-size: 1.2em;
                font-weight: 600;
                margin-bottom: 10px;
            }

            .drop-zone-hint {
                color: #999;
                font-size: 0.9em;
            }

            #fileInput {
                display: none;
            }

            .file-info {
                background: #f0f0f0;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                display: none;
            }

            .file-info.active {
                display: block;
            }

            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 10px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
                margin-top: 20px;
            }

            .btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }

            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }

            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }

            .loading.active {
                display: block;
            }

            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .result {
                background: #f8f9ff;
                padding: 25px;
                border-radius: 15px;
                margin-top: 20px;
                display: none;
                border-left: 5px solid #667eea;
            }

            .result.active {
                display: block;
                animation: slideIn 0.5s ease;
            }

            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .result-title {
                color: #667eea;
                font-weight: 600;
                margin-bottom: 15px;
                font-size: 1.2em;
            }

            .result-text {
                color: #333;
                line-height: 1.6;
                font-size: 1.1em;
                word-wrap: break-word;
            }

            .result-meta {
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #ddd;
                color: #666;
                font-size: 0.9em;
            }

            .error {
                background: #ffe0e0;
                color: #d00;
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
                display: none;
                border-left: 5px solid #d00;
            }

            .error.active {
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Transcription Audio</h1>
            <p class="subtitle">Convertissez vos fichiers audio en texte avec Whisper AI</p>

            <div class="drop-zone" id="dropZone">
                <div class="drop-zone-icon">📁</div>
                <div class="drop-zone-text">Glissez-déposez un fichier audio ici</div>
                <div class="drop-zone-hint">ou cliquez pour sélectionner (WAV, MP3)</div>
            </div>

            <input type="file" id="fileInput" accept=".wav,.mp3" />

            <div class="file-info" id="fileInfo">
                <strong>Fichier sélectionné :</strong> <span id="fileName"></span>
            </div>

            <button class="btn" id="transcribeBtn" disabled>Transcrire</button>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Transcription en cours...</p>
            </div>

            <div class="error" id="error"></div>

            <div class="result" id="result">
                <div class="result-title">📝 Transcription</div>
                <div class="result-text" id="resultText"></div>
                <div class="result-meta">
                    <strong>Fichier :</strong> <span id="resultFile"></span><br>
                    <strong>Langue détectée :</strong> <span id="resultLang"></span>
                </div>
            </div>
        </div>

        <script>
            const dropZone = document.getElementById('dropZone');
            const fileInput = document.getElementById('fileInput');
            const fileInfo = document.getElementById('fileInfo');
            const fileName = document.getElementById('fileName');
            const transcribeBtn = document.getElementById('transcribeBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const error = document.getElementById('error');
            const resultText = document.getElementById('resultText');
            const resultFile = document.getElementById('resultFile');
            const resultLang = document.getElementById('resultLang');

            let selectedFile = null;

            // Click sur la zone de drop
            dropZone.addEventListener('click', () => fileInput.click());

            // Drag and drop events
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });

            dropZone.addEventListener('dragleave', () => {
                dropZone.classList.remove('dragover');
            });

            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            });

            // Sélection de fichier
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFile(e.target.files[0]);
                }
            });

            function handleFile(file) {
                const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3'];
                const validExtensions = ['.wav', '.mp3'];
                const fileExtension = '.' + file.name.split('.').pop().toLowerCase();

                if (!validExtensions.includes(fileExtension)) {
                    showError('Format non supporté. Utilisez WAV ou MP3.');
                    return;
                }

                selectedFile = file;
                fileName.textContent = file.name;
                fileInfo.classList.add('active');
                transcribeBtn.disabled = false;
                result.classList.remove('active');
                error.classList.remove('active');
            }

            // Transcription
            transcribeBtn.addEventListener('click', async () => {
                if (!selectedFile) return;

                const formData = new FormData();
                formData.append('audio_file', selectedFile);

                transcribeBtn.disabled = true;
                loading.classList.add('active');
                result.classList.remove('active');
                error.classList.remove('active');

                try {
                    const response = await fetch('/transcribe', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || 'Erreur lors de la transcription');
                    }

                    // Afficher les résultats
                    resultText.textContent = data.text;
                    resultFile.textContent = data.filename;
                    resultLang.textContent = data.language.toUpperCase();
                    result.classList.add('active');

                } catch (err) {
                    showError(err.message);
                } finally {
                    loading.classList.remove('active');
                    transcribeBtn.disabled = false;
                }
            });

            function showError(message) {
                error.textContent = '❌ ' + message;
                error.classList.add('active');
                setTimeout(() => {
                    error.classList.remove('active');
                }, 5000);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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
            return JSONResponse(
                content={
                    "filename": audio_file.filename,
                    "text": result["text"],
                    "language": result.get("language", "unknown"),
                }
            )
        finally:
            # Nettoyer le fichier temporaire
            os.unlink(temp_path)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la transcription: {str(e)}"
        )
