import argparse
import gc
import json
import os
import re
import warnings
from contextlib import contextmanager
from threading import Lock
from typing import Generator, Optional, Tuple

import torch
import whisper
from peft import PeftModel
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

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a"}


class ModelShifter:
    def __init__(self, whisper_size: str, qwen_adapter_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_available = self.device.type == "cuda"

        self.whisper_size = whisper_size
        self.qwen_adapter_dir = qwen_adapter_dir
        self._lock = Lock()

        self.whisper_model = None
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.loaded_model_name = None

        if self.gpu_available:
            print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
            print(f"Nombre de GPUs: {torch.cuda.device_count()}")
        else:
            print("GPU non disponible, utilisation du CPU")

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

    def _cleanup_memory(self) -> None:
        gc.collect()
        if self.gpu_available:
            torch.cuda.empty_cache()

    @staticmethod
    def _extract_assistant_prediction(generated_text: str) -> str:
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

    def _ensure_whisper_loaded(self) -> None:
        if self.whisper_model is not None:
            return

        startup_device = "cpu"
        print(f"Chargement du modele Whisper {self.whisper_size} sur {startup_device}")
        self.whisper_model = whisper.load_model(
            self.whisper_size, device=startup_device
        )
        print("Modele Whisper charge avec succes")

    def _offload_whisper(self) -> None:
        if self.whisper_model is not None and self.gpu_available:
            self.whisper_model.to(torch.device("cpu"))
        self._cleanup_memory()

    def _move_whisper_to_gpu(self) -> None:
        if self.gpu_available and self.whisper_model is not None:
            self.whisper_model.to(self.device)

    def _ensure_qwen_loaded(self) -> None:
        if self.qwen_model is not None:
            return

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

        self.qwen_model = PeftModel.from_pretrained(base_model, self.qwen_adapter_dir)
        self.qwen_model.eval()

        if self.qwen_tokenizer is None:
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                self.qwen_adapter_dir,
                trust_remote_code=True,
            )
            if self.qwen_tokenizer.pad_token is None and self.qwen_tokenizer.eos_token:
                self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        print("Modele Qwen charge avec succes")

    def _unload_qwen(self) -> None:
        if self.qwen_model is not None:
            del self.qwen_model
            self.qwen_model = None
        self._cleanup_memory()

    @contextmanager
    def use_whisper(self) -> Generator:
        with self._lock:
            if self.loaded_model_name != "whisper":
                if self.loaded_model_name == "qwen":
                    self._unload_qwen()
                self._ensure_whisper_loaded()

            self._move_whisper_to_gpu()
            self.loaded_model_name = "whisper"
            yield self.whisper_model

    @contextmanager
    def use_qwen(self) -> Generator[Tuple, None, None]:
        with self._lock:
            if self.loaded_model_name != "qwen":
                if self.loaded_model_name == "whisper":
                    self._offload_whisper()
                self._ensure_qwen_loaded()

            self.loaded_model_name = "qwen"
            yield self.qwen_model, self.qwen_tokenizer

    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> dict:
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Format non supporte: {ext}. Formats acceptes: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Fichier introuvable: {audio_path}")

        with self.use_whisper() as whisper_model:
            kwargs = {}
            if language:
                kwargs["language"] = language
            result = whisper_model.transcribe(audio_path, **kwargs)

        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", []),
        }

    def generate_latex(self, text: str, max_new_tokens: int = 256) -> str:
        text_input = text.strip()
        if not text_input:
            raise ValueError("Le texte de transcription est vide")

        with self.use_qwen() as (model, tokenizer):
            messages = [
                {
                    "role": "system",
                    "content": "Tu es un assistant mathematique qui traduit de l'anglais parle vers des equations.",
                },
                {
                    "role": "user",
                    "content": text_input,
                },
            ]

            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (AttributeError, ValueError, KeyError, RuntimeError):
                prompt = (
                    "Convertis ce texte mathematique en code LaTeX. "
                    "Reponds uniquement avec du LaTeX.\n\n"
                    f"Texte:\n{text_input}\n\nLaTeX:\n"
                )

            model_inputs = tokenizer(prompt, return_tensors="pt")
            model_device = self._get_model_device(model)
            model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}
            input_length = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = generated[:, input_length:]
            generated_text = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            latex_code = self._extract_assistant_prediction(generated_text)

            if not latex_code:
                raise RuntimeError("La generation LaTeX a retourne une reponse vide")

            return latex_code

    def transcribe_to_latex(
        self,
        audio_path: str,
        max_new_tokens: int = 256,
        language: Optional[str] = None,
    ) -> dict:
        transcription = self.transcribe_audio(audio_path, language=language)
        latex = self.generate_latex(
            transcription["text"], max_new_tokens=max_new_tokens
        )
        return {
            "audio_file": audio_path,
            "language": transcription["language"],
            "text": transcription["text"],
            "latex": latex,
        }


def default_adapter_dir() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "qwen-mathbridge-qlora", "final"))


def list_wav_files(directory_path: str) -> list[str]:
    wav_files = []
    for file_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, file_name)
        if os.path.isfile(full_path) and file_name.lower().endswith(".wav"):
            wav_files.append(full_path)
    wav_files.sort()
    return wav_files


def resolve_output_json_path(input_path: str, output_json: Optional[str]) -> str:
    if output_json:
        return os.path.abspath(output_json)

    if os.path.isdir(input_path):
        return os.path.join(input_path, "transcriptions_latex.json")

    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(
        os.path.dirname(input_path),
        f"{stem}_transcription_latex.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcription audio (Whisper) + conversion en LaTeX (Qwen LoRA)."
    )
    parser.add_argument(
        "input_path",
        help="Chemin vers un fichier audio ou un dossier contenant des fichiers .wav",
    )
    parser.add_argument(
        "output_json",
        help=(
            "Chemin de sortie JSON. Si absent et input_path est un dossier, "
            "le fichier transcriptions_latex.json est cree dans ce dossier."
        ),
    )
    parser.add_argument(
        "--adapter-dir",
        default=default_adapter_dir(),
        help="Chemin vers le dossier adaptateur LoRA Qwen",
    )
    parser.add_argument(
        "--model-size",
        default="turbo",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Taille du modele Whisper",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Nombre max de tokens generes pour la sortie LaTeX",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Langue forcee pour Whisper (ex: fr, en). Laisser vide pour auto-detection.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        input_path = os.path.abspath(args.input_path)

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Chemin introuvable: {input_path}")

        model_shifter = ModelShifter(
            whisper_size=args.model_size,
            qwen_adapter_dir=os.path.abspath(args.adapter_dir),
        )

        output_rows = []

        if os.path.isdir(input_path):
            wav_files = list_wav_files(input_path)
            if not wav_files:
                raise FileNotFoundError(
                    f"Aucun fichier .wav trouve dans le dossier: {input_path}"
                )

            print(f"Traitement de {len(wav_files)} fichier(s) .wav...")
            for audio_path in wav_files:
                print(f"\n--- {os.path.basename(audio_path)} ---")
                result = model_shifter.transcribe_to_latex(
                    audio_path=audio_path,
                    max_new_tokens=args.max_new_tokens,
                    language=args.language,
                )

                output_rows.append(
                    {
                        "nom_fichier_audio": os.path.basename(audio_path),
                        "transcription": result["text"],
                        "Latex": result["latex"],
                    }
                )

                print("Transcription:")
                print(result["text"])
                print("LaTeX:")
                print(result["latex"])

            output_json_path = resolve_output_json_path(input_path, args.output_json)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(output_rows, f, ensure_ascii=False, indent=2)

            print(f"\nResultats JSON ecrits dans: {output_json_path}")
        else:
            result = model_shifter.transcribe_to_latex(
                audio_path=input_path,
                max_new_tokens=args.max_new_tokens,
                language=args.language,
            )

            print("\n=== Transcription ===")
            print(result["text"])
            print("\n=== LaTeX ===")
            print(result["latex"])

            output_rows.append(
                {
                    "nom_fichier_audio": os.path.basename(input_path),
                    "transcription": result["text"],
                    "Latex": result["latex"],
                }
            )

            if args.output_json:
                output_json_path = resolve_output_json_path(
                    input_path, args.output_json
                )
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(output_rows, f, ensure_ascii=False, indent=2)
                print(f"\nResultat JSON ecrit dans: {output_json_path}")

        return 0

    except RECOVERABLE_ERRORS + (FileNotFoundError,) as e:
        print(f"Erreur: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
