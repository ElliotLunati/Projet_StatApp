import gc
import json
import os
import re
import tempfile
import warnings

import torch
import whisper
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore")

RECOVERABLE_ERRORS = (
    RuntimeError,
    ValueError,
    OSError,
    TypeError,
    KeyError,
    torch.cuda.OutOfMemoryError,
)


MODEL_SIZE = "turbo"  # tiny, base, small, medium, large, turbo


class TextToLatexRequest(BaseModel):
    text: str
    max_new_tokens: int = Field(default=256, ge=32, le=1024)


class ModelShifter:
    """Charge les modeles Whisper et Qwen au demarrage et les conserve en memoire."""

    def __init__(self, whisper_size: str, qwen_adapter_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_available = self.device.type == "cuda"

        self.whisper_size = whisper_size
        self.qwen_adapter_dir = qwen_adapter_dir

        if self.gpu_available:
            print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
            print(f"Nombre de GPUs: {torch.cuda.device_count()}")
        else:
            print("GPU non disponible, utilisation du CPU")

        # Charger les modeles immediatement au demarrage
        print("Initialisation des modeles...")
        self.whisper_model = self._load_whisper()
        self.qwen_model, self.qwen_tokenizer = self._load_qwen()
        print("Tous les modeles charges avec succes!")

    @staticmethod
    def _read_base_model_name(adapter_dir: str) -> str:
        adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
        default_base = "Qwen/Qwen3.5-4B"

        try:
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return config.get("base_model_name_or_path", default_base)
        except (OSError, json.JSONDecodeError):
            return default_base

    @staticmethod
    def _extract_assistant_prediction(generated_text: str) -> str:
        """Nettoie la sortie du modele (balises think/chat) pour ne garder que la reponse utile."""
        if not generated_text:
            return ""

        text = generated_text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.replace("<think>", "").replace("</think>", "")
        text = text.replace("<|im_end|>", "").strip()

        if "assistant\\n" in text:
            text = text.split("assistant\\n")[-1].strip()
        if "assistant\n" in text:
            text = text.split("assistant\n")[-1].strip()
        text = re.sub(r"^assistant\s*:?\s*", "", text, flags=re.IGNORECASE)

        if text.startswith("```"):
            text = text.replace("```latex", "").replace("```", "").strip()

        return text

    @staticmethod
    def _get_model_device(model) -> torch.device:
        return next(model.parameters()).device

    def _load_whisper(self):
        """Charge le modele Whisper et le retourne."""
        startup_device = "cpu"
        print(f"Chargement du modele Whisper {self.whisper_size} sur {startup_device}")
        model = whisper.load_model(self.whisper_size, device=startup_device)
        print("Modele Whisper charge avec succes")
        return model

    def _load_qwen(self):
        """Charge le modele Qwen avec son tokenizer et les retourne."""
        base_model_name = self._read_base_model_name(self.qwen_adapter_dir)
        print(f"Chargement du modele Qwen base={base_model_name} avec QLoRA")

        if self.gpu_available:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to(self.device)

        model = PeftModel.from_pretrained(base_model, self.qwen_adapter_dir)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self.qwen_adapter_dir,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token

        print("Modele Qwen charge avec succes")
        return model, tokenizer

    def transcribe(self, audio_path: str) -> dict:
        """Transcrit un fichier audio et retourne le resultat."""
        return self.whisper_model.transcribe(audio_path)

    def generate_latex(self, text: str, max_new_tokens: int = 256) -> str:
        """Genere du code LaTeX a partir du texte en utilisant Qwen."""
        # Prompt aligne avec le schema d'inference valide dans evaluation.py.
        messages = [
            {
                "role": "system",
                "content": "Tu es un assistant mathematique qui traduit de l'anglais parle vers des equations.",
            },
            {"role": "user", "content": text},
        ]

        try:
            prompt = self.qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, ValueError, KeyError, RuntimeError):
            prompt = (
                "Convertis ce texte mathematique en code LaTeX."
                "Reponds uniquement avec du LaTeX.\n\n"
                f"Texte:\n{text}\n\nLaTeX:\n"
            )

        model_inputs = self.qwen_tokenizer(prompt, return_tensors="pt")

        model_device = self._get_model_device(self.qwen_model)
        model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}
        input_length = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generated = self.qwen_model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.qwen_tokenizer.pad_token_id,
                eos_token_id=self.qwen_tokenizer.eos_token_id,
            )

        generated_ids = generated[:, input_length:]
        generated_text = self.qwen_tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        latex_code = self._extract_assistant_prediction(generated_text)

        if not latex_code:
            raise RuntimeError("La generation LaTeX a retourne une reponse vide")

        return latex_code


app = FastAPI()

app.mount(
    "/images",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "images")),
    name="images",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
qwen_adapter_candidates = [
    os.path.join(current_dir, "qwen-mathbridge-qlora", "final"),
    os.path.join(current_dir, "..", "qwen-mathbridge-qlora", "final"),
]
qwen_adapter_path = next(
    (
        os.path.abspath(candidate)
        for candidate in qwen_adapter_candidates
        if os.path.isdir(candidate)
    ),
    os.path.abspath(qwen_adapter_candidates[0]),
)

model_shifter = ModelShifter(
    whisper_size=MODEL_SIZE,
    qwen_adapter_dir=qwen_adapter_path,
)


@app.get("/")
def read_root():
    """Interface web pour uploader et transcrire des fichiers audio"""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    return FileResponse(template_path)


@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
):
    """
    Reçoit un fichier audio, retourne la transcription texte et la conversion LaTeX.
    """
    allowed_extensions = [".wav", ".mp3", ".webm", ".ogg", ".m4a"]
    file_extension = os.path.splitext(audio_file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté. Formats acceptés: {', '.join(allowed_extensions)}",
        )

    try:
        print(f"Transcription de {audio_file.filename}...")

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            result = model_shifter.transcribe(temp_path)

            segments = result.get("segments", [])
            for seg in segments:
                print(
                    f"[{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text'].strip()}"
                )

            transcription_text = result["text"].strip()
            if not transcription_text:
                raise RuntimeError("Aucun texte detecte dans l'audio")

            latex_code = model_shifter.generate_latex(transcription_text)

            return JSONResponse(
                content={
                    "filename": audio_file.filename,
                    "text": transcription_text,
                    "language": result.get("language", "unknown"),
                    "latex": latex_code,
                }
            )
        finally:
            # Nettoyer le fichier temporaire
            os.unlink(temp_path)

    except RECOVERABLE_ERRORS as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la transcription: {str(e)}"
        ) from e


@app.post("/text-to-latex")
async def text_to_latex(payload: TextToLatexRequest):
    text_input = payload.text.strip()
    if not text_input:
        raise HTTPException(status_code=400, detail="Le champ texte est vide")

    try:
        latex_code = model_shifter.generate_latex(
            text=text_input,
            max_new_tokens=payload.max_new_tokens,
        )
        return JSONResponse(
            content={
                "input_text": payload.text,
                "latex": latex_code,
            }
        )
    except RECOVERABLE_ERRORS as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la generation LaTeX: {str(e)}",
        ) from e
