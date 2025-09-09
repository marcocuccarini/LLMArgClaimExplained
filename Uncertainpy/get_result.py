import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_graphs(base_dir):
    results = []
    all_claim_true, all_claim_pred = [], []
    all_edge_true, all_edge_pred = [], []

    # Store pairs for inspection
    claim_pairs = []
    edge_pairs = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if re.match(r"(graphs_.*\.json|predictions_.*\.json)", file):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Case 1: Edge-level evaluation
                    if isinstance(data, dict) and "edges" in data:
                        y_true, y_pred = [], []
                        for edge in data.get("edges", []):
                            if "gold_relation" in edge and "relation" in edge:
                                y_true.append(edge["gold_relation"])
                                y_pred.append(edge["relation"])
                                edge_pairs.append({
                                    "file": file,
                                    "source": edge.get("source"),
                                    "target": edge.get("target"),
                                    "gold": edge["gold_relation"],
                                    "pred": edge["relation"]
                                })

                        if y_true:
                            acc = accuracy_score(y_true, y_pred)
                            f1 = f1_score(y_true, y_pred, average="weighted")
                            results.append({
                                "directory": os.path.basename(root),
                                "file": file,
                                "level": "edge",
                                "accuracy": acc,
                                "f1_score": f1
                            })
                            all_edge_true.extend(y_true)
                            all_edge_pred.extend(y_pred)

                    # Case 2: Claim-level evaluation
                    elif isinstance(data, list):
                        y_true, y_pred = [], []
                        for item in data:
                            if "ground_truth" in item and "prediction" in item:
                                y_true.append(item["ground_truth"])
                                y_pred.append(item["prediction"])

                                # Find strongest evidence if graph is available
                                strongest_evidence = None
                                strongest_strength = None
                                claim_strength = None
                                if "graph" in item and "nodes" in item["graph"]:
                                    nodes = item["graph"]["nodes"]

                                    # strongest evidence
                                    evidence_nodes = [n for n in nodes if n.get("type") == "evidence"]
                                    if evidence_nodes:
                                        strongest = max(
                                            evidence_nodes,
                                            key=lambda x: x.get("strength", 0)
                                        )
                                        strongest_evidence = strongest.get("text", "")
                                        strongest_strength = strongest.get("strength", None)

                                    # claim strength
                                    claim_nodes = [n for n in nodes if n.get("type") == "claim"]
                                    if claim_nodes:
                                        claim_strength = claim_nodes[0].get("strength", None)

                                claim_pairs.append({
                                    "file": file,
                                    "claim": item.get("claim", ""),
                                    "gold": item["ground_truth"],
                                    "pred": item["prediction"],
                                    "claim_strength": claim_strength,
                                    "strongest_evidence": strongest_evidence,
                                    "strongest_strength": strongest_strength
                                })

                        if y_true:
                            acc = accuracy_score(y_true, y_pred)
                            f1 = f1_score(y_true, y_pred, average="weighted")
                            results.append({
                                "directory": os.path.basename(root),
                                "file": file,
                                "level": "claim",
                                "accuracy": acc,
                                "f1_score": f1
                            })
                            all_claim_true.extend(y_true)
                            all_claim_pred.extend(y_pred)

                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")

    # Convert results to DataFrames
    df = pd.DataFrame(results)
    df_claim_pairs = pd.DataFrame(claim_pairs)
    df_edge_pairs = pd.DataFrame(edge_pairs)

    if not df.empty:
        # Save results
        df.to_csv("evaluation_results.csv", index=False)

        # Save pairs
        if not df_claim_pairs.empty:
            df_claim_pairs.to_csv("all_claim_predictions.csv", index=False)
        if not df_edge_pairs.empty:
            df_edge_pairs.to_csv("all_edge_predictions.csv", index=False)

        # Correlation matrix
        corr_matrix = df[["accuracy", "f1_score"]].corr()
        corr_matrix.to_csv("evaluation_correlation_matrix.csv")

        # Confusion matrix (claim-level)
        if all_claim_true:
            cm = confusion_matrix(all_claim_true, all_claim_pred, labels=list(set(all_claim_true)))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=list(set(all_claim_true)))
            disp.plot(cmap="Blues", xticks_rotation=45)
            plt.title("Confusion Matrix (Claim-level)")
            plt.tight_layout()
            plt.savefig("confusion_matrix_claim.png", dpi=300)
            plt.show()

        # Confusion matrix (edge-level)
        if all_edge_true:
            cm = confusion_matrix(all_edge_true, all_edge_pred, labels=list(set(all_edge_true)))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=list(set(all_edge_true)))
            disp.plot(cmap="Purples", xticks_rotation=45)
            plt.title("Confusion Matrix (Edge-level)")
            plt.tight_layout()
            plt.savefig("confusion_matrix_edge.png", dpi=300)
            plt.show()

    else:
        print("⚠️ No evaluation results found")
        corr_matrix = pd.DataFrame()

    return df, df_claim_pairs, df_edge_pairs, corr_matrix


# Example usage:
base_directory = "results"
df_results, df_claim_pairs, df_edge_pairs, corr = evaluate_graphs(base_directory)

print("\nResults DataFrame:\n", df_results)
print("\nClaim Predictions vs Ground Truth (+ claim strength + strongest evidence):\n", df_claim_pairs.head())
print("\nEdge Predictions vs Ground Truth:\n", df_edge_pairs.head())
