import torch
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

try:
    import editdistance  # type: ignore[import-not-found]
except ImportError:
    editdistance = None

# --- CONFIGURATION ---
BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
ADAPTER_PATH = "./qwen-mathbridge-qlora/final"
DATASET_ID = "Kyudan/MathBridge"
DATASET_SPLIT = "train"
SHUFFLE_SEED = 42
NUM_EVAL_SAMPLES = 100


def clean_latex(latex_str):
    """Nettoie les espaces pour une comparaison plus juste."""
    if not latex_str:
        return ""
    # Supprime les espaces multiples et les espaces autour des opérateurs basiques
    latex_str = re.sub(r"\s+", "", latex_str)
    return latex_str.strip()


def calculate_cer(reference, hypothesis):
    """Calcule le CER via distance de Levenshtein / longueur de la référence."""
    if reference is None:
        reference = ""
    if hypothesis is None:
        hypothesis = ""

    reference = str(reference)
    hypothesis = str(hypothesis)

    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    distance = levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def levenshtein_distance(reference, hypothesis):
    """Calcule la distance de Levenshtein (fallback natif si editdistance absent)."""
    if editdistance is not None:
        return editdistance.eval(reference, hypothesis)

    if reference == hypothesis:
        return 0
    if len(reference) == 0:
        return len(hypothesis)
    if len(hypothesis) == 0:
        return len(reference)

    previous_row = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current_row = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (ref_char != hyp_char)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row

    return previous_row[-1]


def load_evaluation_samples():
    """Charge, mélange et échantillonne le dataset d'évaluation."""
    dataset = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    dataset = dataset.shuffle(seed=SHUFFLE_SEED)
    sample_count = min(NUM_EVAL_SAMPLES, len(dataset))
    return dataset.select(range(sample_count))


def extract_assistant_prediction(generated_text):
    """Nettoie la sortie du modèle et retire les balises de raisonnement."""
    if not generated_text:
        return ""

    text = generated_text.strip()

    # Retire les blocs <think>...</think> sur plusieurs lignes.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")

    # Nettoie d'éventuels marqueurs ChatML restants.
    text = text.replace("<|im_end|>", "").strip()

    # Si la réponse contient encore un préfixe assistant, on garde la fin utile.
    if "assistant\\n" in text:
        text = text.split("assistant\\n")[-1].strip()
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1].strip()
    text = re.sub(r"^assistant\s*:?\s*", "", text, flags=re.IGNORECASE)

    return text


def generate_prediction(model, tokenizer, prompt):
    """Génère une prédiction nettoyée à partir d'un prompt ChatML."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return extract_assistant_prediction(generated_text)


def main():

    print(f"Chargement du dataset {DATASET_ID} (split={DATASET_SPLIT})...")
    eval_samples = load_evaluation_samples()

    print("Chargement du modèle en 4-bit pour l'évaluation...")

    # On garde le 4-bit pour l'inférence pour être sûr que ça passe sur les 8 Go de la RTX 3070
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )

    print("Branchement de l'adaptateur LoRA...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()  # Passage en mode évaluation

    total = len(eval_samples)
    evaluated = 0
    base_exact_matches = 0
    finetuned_exact_matches = 0
    base_cer_sum = 0.0
    finetuned_cer_sum = 0.0

    print("\nDébut de l'évaluation :\n" + "-" * 40)

    for i, sample in enumerate(eval_samples, start=1):
        spoken = sample.get("spoken_English", "")
        expected = sample.get("equation", "")

        if not spoken or not expected:
            print(f"Exemple {i}/{total} ignoré (champs manquants).")
            print("-" * 40)
            continue

        evaluated += 1

        # Recréer le prompt exact (ChatML)
        messages = [
            {
                "role": "system",
                "content": "Tu es un assistant mathematique qui traduit de l'anglais parle vers des equations.",
            },
            {"role": "user", "content": spoken},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Prédiction avec adaptateur LoRA actif (modèle fine-tuné).
        finetuned_prediction = generate_prediction(model, tokenizer, prompt)

        # Prédiction du modèle de base (non fine-tuné) en désactivant temporairement LoRA.
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                base_prediction = generate_prediction(model, tokenizer, prompt)
        elif hasattr(model, "get_base_model"):
            base_prediction = generate_prediction(
                model.get_base_model(), tokenizer, prompt
            )
        else:
            base_prediction = "(indisponible: impossible de désactiver l'adaptateur)"

        expected_clean = clean_latex(expected)
        base_prediction_clean = clean_latex(base_prediction)
        finetuned_prediction_clean = clean_latex(finetuned_prediction)

        base_exact = base_prediction_clean == expected_clean
        finetuned_exact = finetuned_prediction_clean == expected_clean

        base_cer = calculate_cer(expected_clean, base_prediction_clean)
        finetuned_cer = calculate_cer(expected_clean, finetuned_prediction_clean)

        if base_exact:
            base_exact_matches += 1
        if finetuned_exact:
            finetuned_exact_matches += 1

        base_cer_sum += base_cer
        finetuned_cer_sum += finetuned_cer

        # Affichage des résultats
        print(f"Exemple {i}/{total}")
        print(f"Entrée (Anglais) : {spoken}")
        print(f"Attendu                                      : {expected}")
        print(f"Prédiction (Qwen3.5-4b base)                 : {base_prediction}")
        print(f"CER base                                     : {base_cer:.4f}")
        print(
            f"Exact Match base                             : {'oui' if base_exact else 'non'}"
        )
        print(f"Prédiction (Qwen3.5-4b-mathsbridge-qlora)    : {finetuned_prediction}")
        print(f"CER fine-tuné                                : {finetuned_cer:.4f}")
        print(
            f"Exact Match fine-tuné                        : {'oui' if finetuned_exact else 'non'}"
        )
        print("-" * 40)

    if evaluated == 0:
        print("Aucun exemple valide n'a pu être évalué.")
        return

    base_exact_pct = (base_exact_matches / evaluated) * 100
    finetuned_exact_pct = (finetuned_exact_matches / evaluated) * 100
    base_avg_cer = base_cer_sum / evaluated
    finetuned_avg_cer = finetuned_cer_sum / evaluated

    print("\nRésumé global :\n" + "=" * 40)
    print(f"Exemples évalués : {evaluated}/{total}")
    print(f"Exact Match base (%)                          : {base_exact_pct:.2f}")
    print(f"Exact Match fine-tuné (%)                     : {finetuned_exact_pct:.2f}")
    print(f"CER moyen base                                : {base_avg_cer:.4f}")
    print(f"CER moyen fine-tuné                           : {finetuned_avg_cer:.4f}")


if __name__ == "__main__":
    main()
