import os
import json
import pickle
import re
from huggingface_hub import hf_hub_download
import ollama
from classes.Ollama import ensure_ollama_model, run_ollama_inference
from classes.prompt import FACTCHECK_PROMPTS

# === Parse predictions ===
def extract_first_true_false(text):
    if not text:
        return None
    m = re.search(r'\b(true|false)\b', text, re.IGNORECASE)
    if m:
        return "TRUE" if m.group(1).lower() == "true" else "FALSE"
    return None

def parse_prediction(answer, prompt_type):
    if not answer:
        return "INVALID"
    if prompt_type == "binary":
        tf = extract_first_true_false(answer)
        return tf if tf else "INVALID"
    return "INVALID"

# === Master summary updater ===
def update_master_summary(new_summary, master_summary_file="results/master_summary.json"):
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

# === Experiment Runner ===
def run_experiment(K, MODEL, prompt_name, prompt_template, prompt_type, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    model_safe = MODEL.replace(":", "-").replace("/", "-")
    predictions_file = os.path.join(save_dir, f"predictions_{prompt_name}_k{K}_{model_safe}.json")

    file_path = hf_hub_download("Yuqicheng/ArgRAG", "PubHealth.pkl", repo_type="dataset")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    ensure_ollama_model(MODEL)

    results = []
    for idx, (claim, contexts, gt) in enumerate(zip(data["claims"], data["contexts"], data["answers"])):
        cur_contexts = contexts[:min(K, len(contexts))]

        # Run prompt
        evidence_text = "\n".join(f"- {ctx}" for ctx in cur_contexts)
        prompt = prompt_template.format(claim=claim, evidence=evidence_text, example="{}")
        answer = run_ollama_inference(prompt, model=MODEL)
        pred = parse_prediction(answer, prompt_type)

        results.append({
            "index": idx,
            "claim": claim,
            "prediction": pred,
            "ground_truth": gt.strip().upper(),
            "model": MODEL,
            "K": K,
            "prompt": prompt_name
        })

        # Save periodically
        if idx % 10 == 0 and idx > 0:
            with open(predictions_file, "w") as f:
                json.dump(results, f, indent=2)

    # Final save
    with open(predictions_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save summary (without metrics)
    summary = [{"prompt": prompt_name, "model": MODEL, "K": K}]
    summary_file = os.path.join(save_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Update master summary
    update_master_summary(summary, master_summary_file="results/master_summary.json")

    print(f"Finished: {prompt_name} | {MODEL} | K={K}")

    return predictions_file, results

# === Main Loop ===
if __name__ == "__main__":
    Ks = [5, 10]
    MODELS = ["gemma3:27b"]
    save_dir = "results"
    for K in Ks:
        for MODEL in MODELS:
            for prompt_name, (template, ptype) in FACTCHECK_PROMPTS.items():
                run_experiment(K, MODEL, prompt_name, template, ptype, save_dir=save_dir)
