import os
import json
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from huggingface_hub import hf_hub_download

input_dir = "results"
output_file = "all_predictions_metrics_with_gt.json"
valid_labels = {"TRUE", "FALSE"}  # Only these are usable

# === Choose which families of models to print ===
# Example: ["gemma3", "gpt-oss"], or [] to print all
MODEL_FAMILIES_TO_PRINT = ["gpt-oss"]

# === Load dataset for ground truths ===
dataset_file = hf_hub_download("Yuqicheng/ArgRAG", "PubHealth.pkl", repo_type="dataset")
with open(dataset_file, "rb") as f:
    dataset = pickle.load(f)

dataset_answers = [str(ans).strip().upper() if ans else None for ans in dataset["answers"]]

all_metrics = []

for filename in os.listdir(input_dir):
    if filename.startswith("predictions") and filename.endswith(".json"):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        # Assign ground truth by dataset index
        for item in data:
            idx = item["index"]
            item["ground_truth"] = dataset_answers[idx]

        total_items = len(data)

        # Split usable vs invalid predictions
        filtered = [
            item for item in data
            if item.get("prediction") in valid_labels and item.get("ground_truth") in valid_labels
        ]
        invalid = [item for item in data if item.get("prediction") not in valid_labels]

        if not filtered:
            continue

        predictions = [item["prediction"] for item in filtered]
        ground_truths = [item["ground_truth"] for item in filtered]

        # Metrics
        accuracy = accuracy_score(ground_truths, predictions)
        f1 = f1_score(ground_truths, predictions, labels=["TRUE", "FALSE"], average='macro')
        cm = confusion_matrix(ground_truths, predictions, labels=["TRUE", "FALSE"])
        cm_dict = {
            "TRUE_pred_TRUE": int(cm[0][0]),
            "TRUE_pred_FALSE": int(cm[0][1]),
            "FALSE_pred_TRUE": int(cm[1][0]),
            "FALSE_pred_FALSE": int(cm[1][1])
        }

        model_name = data[0].get("model", "unknown")
        invalid_rate = len(invalid) / total_items if total_items > 0 else 0.0

        info = {
            "file": filename,
            "model": model_name,
            "K": data[0].get("K", "unknown"),
            "prompt": data[0].get("prompt", "unknown"),
            "accuracy": accuracy,
            "f1_score": f1,
            "confusion_matrix": cm_dict,
            "num_samples": len(filtered),
            "invalid_percentage": round(invalid_rate * 100, 2)
        }
        all_metrics.append(info)

        # === Print only for selected families ===
        if not MODEL_FAMILIES_TO_PRINT or any(model_name.startswith(fam) for fam in MODEL_FAMILIES_TO_PRINT):
            print(f"\n=== Metrics for {filename} ===")
            print("Model:", model_name, "| K:", info["K"], "| Prompt:", info["prompt"])
            print("Samples evaluated:", info["num_samples"], f"(out of {total_items})")
            print("Accuracy:", round(info["accuracy"], 4))
            print("F1 Score (macro):", round(info["f1_score"], 4))
            print("Invalid outputs:", len(invalid), f"({info['invalid_percentage']}%)")
            print("Confusion Matrix:", cm_dict)

# Save metrics
with open(output_file, "w") as f:
    json.dump(all_metrics, f, indent=4)

print(f"\n✅ Metrics saved to {output_file}")
