import os
import json
from sklearn.metrics import accuracy_score, f1_score

directory = "gpt-oss-20b/K5"

all_predictions = []
all_ground_truths = []

for root, _, files in os.walk(directory):
    for filename in files:
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(root, filename)
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON in {filename}")
                continue

            file_predictions = []
            file_ground_truths = []

            for item in data:
                pred = item.get('prediction')
                true = item.get('ground_truth')

                if pred is not None and true is not None:
                    file_predictions.append(pred)
                    file_ground_truths.append(true)
                    all_predictions.append(pred)
                    all_ground_truths.append(true)
                else:
                    print(f"Warning: Missing keys in {filename}, item index {item.get('index')}")

            # Print metrics for this file
            if file_predictions:
                file_accuracy = accuracy_score(file_ground_truths, file_predictions)
                file_f1 = f1_score(file_ground_truths, file_predictions, pos_label="TRUE")
                print(f"{filename} -> Accuracy: {file_accuracy:.2f}, F1 Score: {file_f1:.2f}")

# Print overall metrics
if all_predictions:
    accuracy = accuracy_score(all_ground_truths, all_predictions)
    f1 = f1_score(all_ground_truths, all_predictions, pos_label="TRUE")
    print(f"Overall -> Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")
else:
    print("No valid prediction/ground_truth pairs found.")
