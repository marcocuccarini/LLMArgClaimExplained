import os
import json
import pickle
import re
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from classes.Ollama import ensure_ollama_model, run_ollama_inference
from classes.prompt import FACTCHECK_PROMPTS
from sklearn.metrics import accuracy_score

# === Parse predictions ===
def extract_first_true_false(text):
    if not text:
        return None
    m = re.search(r'\b(true|false|yes|no|correct|incorrect)\b', text, re.IGNORECASE)
    if m:
        word = m.group(1).lower()
        if word in ["true", "yes", "correct"]:
            return "TRUE"
        elif word in ["false", "no", "incorrect"]:
            return "FALSE"
    return None

def parse_prediction(answer, prompt_type):
    if not answer:
        return "INVALID"
    if prompt_type == "binary":
        tf = extract_first_true_false(answer)
        return tf if tf else "INVALID"
    return "INVALID"

# === Master summary updater ===
def update_master_summary(new_summary, master_summary_file="result/master_summary.json"):
    os.makedirs(os.path.dirname(master_summary_file), exist_ok=True)
    if os.path.exists(master_summary_file):
        with open(master_summary_file, "r") as f:
            master = json.load(f)
    else:
        master = []
    existing_keys = {(e.get("model"), e.get("K"), e.get("prompt")) for e in master}
    for e in new_summary:
        key = (e.get("model"), e.get("K"), e.get("prompt"))
        if key not in existing_keys:
            master.append(e)
            existing_keys.add(key)
    with open(master_summary_file, "w") as f:
        json.dump(master, f, indent=2)
    print(f"Master summary updated: {master_summary_file}")

# === Experiment Runner with progress bar and accuracy ===
def run_experiment(K, MODEL, prompt_name, prompt_template, prompt_type, save_dir="result"):
    model_safe = MODEL.replace(":", "-").replace("/", "-")
    k_dir = f"K{K}"
    output_dir = os.path.join(save_dir, model_safe, k_dir)
    os.makedirs(output_dir, exist_ok=True)

    predictions_file = os.path.join(output_dir, f"predictions_{prompt_name}.json")
    summary_file = os.path.join(output_dir, "summary.json")

    # Load dataset
    file_path = hf_hub_download("Yuqicheng/ArgRAG", "PubHealth.pkl", repo_type="dataset")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    ensure_ollama_model(MODEL)
    results = []

    for idx, (claim, contexts, gt) in enumerate(tqdm(zip(data["claims"], data["contexts"], data["answers"]),
                                                     total=len(data["claims"]),
                                                     desc=f"{prompt_name} | {MODEL} | K={K}")):
        cur_contexts = contexts[:min(K, len(contexts))]
        evidence_text = "\n".join(f"- {ctx}" for ctx in cur_contexts)
        prompt = (
            f"Claim: {claim}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Answer only with 'TRUE' or 'FALSE'."
        )
        answer = run_ollama_inference(prompt, model=MODEL)
        pred = parse_prediction(answer, prompt_type)

        results.append({
            "index": idx,
            "claim": claim,
            "prediction": pred,
            "ground_truth": gt.strip().upper(),
            "model": MODEL,
            "K": K,
            "prompt": prompt_name,
            "raw_output": answer.strip() if answer else "[EMPTY RESPONSE]"
        })

        # Save periodically
        if idx % 10 == 0 and idx > 0:
            with open(predictions_file, "w") as f:
                json.dump(results, f, indent=2)

    # Final save
    with open(predictions_file, "w") as f:
        json.dump(results, f, indent=2)

    # Compute accuracy
    y_true = [r["ground_truth"] for r in results if r["prediction"] in ["TRUE", "FALSE"]]
    y_pred = [r["prediction"] for r in results if r["prediction"] in ["TRUE", "FALSE"]]
    accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
    print(f"Accuracy for {prompt_name} | {MODEL} | K={K}: {accuracy:.2%}")

    # Save summary
    summary = [{"prompt": prompt_name, "model": MODEL, "K": K, "accuracy": accuracy}]
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Update master summary
    update_master_summary(summary, master_summary_file=os.path.join(save_dir, "master_summary.json"))

    return predictions_file, results

# === Main Loop ===
if __name__ == "__main__":
    Ks = [5]
    MODELS = ["gpt-oss:20b"]
    save_dir = "result"

    for K in Ks:
        for MODEL in MODELS:
            for prompt_name, (template, ptype) in FACTCHECK_PROMPTS.items():
                run_experiment(K, MODEL, prompt_name, template, ptype, save_dir=save_dir)
