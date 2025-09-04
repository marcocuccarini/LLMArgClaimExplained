import json
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Directory containing prediction files
input_dir = "result_base"  # replace with your directory path
output_file = "all_predictions_metrics.json"

all_metrics = []

# Iterate over all files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):  # only process JSON files
        filepath = os.path.join(input_dir, filename)
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Extract predictions and ground truths
        predictions = [item["prediction"] for item in data]
        ground_truths = [item["ground_truth"] for item in data]
        
        # Compute metrics
        accuracy = accuracy_score(ground_truths, predictions)
        f1 = f1_score(ground_truths, predictions, pos_label="TRUE")
        cm = confusion_matrix(ground_truths, predictions, labels=["TRUE", "FALSE"])
        
        # Convert confusion matrix to a dict for JSON
        cm_dict = {
            "TRUE_TRUE": int(cm[0][0]),
            "TRUE_FALSE": int(cm[0][1]),
            "FALSE_TRUE": int(cm[1][0]),
            "FALSE_FALSE": int(cm[1][1])
        }
        
        # Gather info
        info = {
            "file": filename,
            "model": data[0]["model"],
            "K": data[0]["K"],
            "prompt": data[0]["prompt"],
            "accuracy": accuracy,
            "f1_score": f1,
            "confusion_matrix": cm_dict
        }
        
        all_metrics.append(info)

# Save all metrics to a single JSON file
with open(output_file, "w") as f:
    json.dump(all_metrics, f, indent=4)

print(f"Metrics for all files saved to {output_file}")
