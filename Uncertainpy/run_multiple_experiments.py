import json
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

input_dir = "results"
output_file = "all_predictions_metrics.json"

all_metrics = []

valid_labels = {"TRUE", "FALSE"}

for filename in os.listdir(input_dir):
    if filename.startswith("predictions") and filename.endswith(".json"):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        predictions = [item.get("prediction") for item in data]
        ground_truths = [item.get("ground_truth") for item in data]

        # Print unique values to debug errors
        print(f"\n=== Debug values for {filename} ===")
        print("Unique predictions:", set(predictions))
        print("Unique ground truths:", set(ground_truths))

        # Filter valid labels
        filtered = [item for item in data if item.get("prediction") in valid_labels and item.get("ground_truth") in valid_labels]
        if not filtered:
            print(f"Skipping {filename}: no usable predictions")
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

        info = {
            "file": filename,
            "model": data[0].get("model", "unknown"),
            "K": data[0].get("K", "unknown"),
            "prompt": data[0].get("prompt", "unknown"),
            "accuracy": accuracy,
            "f1_score": f1,
            "confusion_matrix": cm_dict,
            "num_samples": len(filtered)
        }
        all_metrics.append(info)

        # Print metrics
        print(f"\n=== Metrics for {filename} ===")
        print("Samples evaluated:", info["num_samples"])
        print("Accuracy:", round(info["accuracy"], 4))
        print("F1 Score (macro):", round(info["f1_score"], 4))
        print("Confusion Matrix:", cm_dict)

# Save metrics
with open(output_file, "w") as f:
    json.dump(all_metrics, f, indent=4)

print(f"\n✅ Metrics saved to {output_file}")
