import json

# Load data from file
with open("predictions_K5_gpt-oss-20b.json", "r") as f:
    data = json.load(f)

# Counters
correct = 0
total = 0
no_evidence_count = 0

# Evaluate only TRUE/FALSE cases
for item in data:
    prediction = item["prediction"]
    ground_truth = item["ground_truth"]

    if ground_truth in ["TRUE", "FALSE"]:
        total += 1
        if prediction == ground_truth:
            correct += 1
    else:
        no_evidence_count += 1

# Compute accuracy
accuracy = correct / total if total > 0 else 0

total_items = total + no_evidence_count
no_evidence_percent = (no_evidence_count / total_items * 100) if total_items > 0 else 0
overall_accuracy = correct / total_items if total_items > 0 else 0

print(f"Evaluated {total} items (ignoring NO_EVIDENCE)")
print(f"Correct predictions: {correct}")
print(f"Accuracy (without NO_EVIDENCE): {accuracy:.2%}")
print(f"NO_EVIDENCE cases: {no_evidence_count}")
print(f"Total items including NO_EVIDENCE: {total_items}")
print(f"NO_EVIDENCE percentage: {no_evidence_percent:.2f}%")
print(f"Overall accuracy (including NO_EVIDENCE as incorrect): {overall_accuracy:.2%}")