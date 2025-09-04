import os
import json
import pickle
import re
import sys
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import networkx as nx
import ollama

# === Add Uncertainpy to path ===
sys.path.append("src")
import uncertainpy.gradual as grad
from uncertainpy.gradual.Argument import Argument

# === Ollama Inference ===
def run_ollama_inference(prompt, model, temperature=0.0, max_tokens=512):
    try:
        response = ollama.chat(model=model, messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])
        if not response or "message" not in response or "content" not in response["message"]:
            return "{}"
        return response["message"]["content"]
    except Exception as e:
        print(f"Ollama error: {e}")
        return "{}"


# === Prompts ===
CLAIM_EVIDENCE_PROMPT = """
Task: 

Given a claim and one piece of evidence, classify the evidence in relation to the claim.
Choose exactly one: "support", "attack", "not_related".


Instructions:

- Support:  Evidence that backs the claim.
- Contradict:  Evidence that counters or limits the claim.
- Irrelevant:  Evidence unrelated to the claim.

Output Format:

Return one label: "support", "contradict", or "irrelevant".

Claim:
{claim}

Evidence:
{evidence}
"""

EVIDENCE_EVIDENCE_PROMPT = """
Task: 

Given two pieces of evidence, classify the relation of the first to the second.
Choose exactly one: "support", "attack", "not_related".


Instructions:

- Support:  Evidence that backs the claim.
- Contradict:  Evidence that counters or limits the claim.
- Irrelevant:  Evidence unrelated to the claim.

Output Format:

Return one label: "support", "contradict", or "irrelevant".

Evidence A:
{e1}

Evidence B:
{e2}
"""


# === Pairwise classification ===
def classify_claim_evidence(claim, ev_id, ev_text, model_name="llama3.1"):
    prompt = CLAIM_EVIDENCE_PROMPT.format(claim=claim, evidence=ev_text)
    response = run_ollama_inference(prompt, model=model_name).strip().lower()

    if "support" in response:
        rel = "support"
    elif "attack" in response or "contradict" in response:
        rel = "attack"
    else:
        rel = "not_related"

    return ("Claim", ev_id, rel)


def classify_evidence_evidence(ev1_id, ev1_text, ev2_id, ev2_text, model_name="llama3.1"):
    prompt = EVIDENCE_EVIDENCE_PROMPT.format(e1=ev1_text, e2=ev2_text)
    response = run_ollama_inference(prompt, model=model_name).strip().lower()

    if "support" in response:
        rel = "support"
    elif "attack" in response or "contradict" in response:
        rel = "attack"
    else:
        rel = "not_related"

    return (ev1_id, ev2_id, rel)


# === Argumentation Graph Construction ===
def ArgRAG_pred_divided(relations, arg_dict, arg_model):
    if not relations:
        return "use parametric answer", None, {}

    G = nx.DiGraph()
    # Add nodes
    for k, v in arg_dict.items():
        node_type = "claim" if k == "Claim" else "evidence"
        G.add_node(k, type=node_type, text=v)

    # Add edges
    for src, tgt, rel in relations:
        if src in arg_dict and tgt in arg_dict:
            if rel == "support":
                G.add_edge(src, tgt, relation="support")
            elif rel == "attack":
                G.add_edge(src, tgt, relation="attack")

    # Build BAG
    bag = grad.BAG()
    for n in G.nodes:
        bag.arguments[n] = Argument(n, 0.5)

    for u, v, d in G.edges(data=True):
        if d["relation"] == "support":
            bag.add_support(bag.arguments[u], bag.arguments[v])
        elif d["relation"] == "attack":
            bag.add_attack(bag.arguments[u], bag.arguments[v])

    arg_model.BAG = bag
    arg_model.approximator = grad.algorithms.RK4(arg_model)
    arg_model.solve(delta=1e-2, epsilon=1e-4)

    strengths = {a.name: a.strength for a in bag.arguments.values()}
    nx.set_node_attributes(G, strengths, "strength")

    return ("true" if strengths["Claim"] >= 0.5 else "false"), G, strengths


# === Prediction ===
def run_arg_rag_prediction(claim, contexts, K, model_name="llama3.1"):
    contexts_to_use = contexts[:min(K, len(contexts))]
    arg_dict = {f"E{i+1}": ctx for i, ctx in enumerate(contexts_to_use)}
    arg_dict["Claim"] = claim

    arg_model = grad.semantics.ContinuousDFQuADModel()
    relations = []
    evidences = [f"E{i+1}" for i in range(len(contexts_to_use))]

    # Claim vs Evidence
    for ev in evidences:
        rel = classify_claim_evidence(claim, ev, arg_dict[ev], model_name)
        if rel[2] != "not_related":
            relations.append(rel)

    # Evidence vs Evidence (directed)
    for i, e1 in enumerate(evidences):
        for j, e2 in enumerate(evidences):
            if i == j:
                continue
            rel = classify_evidence_evidence(e1, arg_dict[e1], e2, arg_dict[e2], model_name)
            if rel[2] != "not_related":
                relations.append(rel)

    pred, G, strengths = ArgRAG_pred_divided(relations, arg_dict, arg_model)

    if G is None or len(G.nodes) == 0:
        return "NO_EVIDENCE", {"nodes": None, "edges": None}

    graph_json = {
        "nodes": [
            {"id": n,
             "type": G.nodes[n].get("type", "unknown"),
             "strength": G.nodes[n].get("strength", 0.5),
             "text": arg_dict[n]}
            for n in G.nodes
        ],
        "edges": [{"source": u, "target": v, "relation": d.get("relation", "unknown")}
                  for u, v, d in G.edges(data=True)]
    }

    return "TRUE" if pred == "true" else "FALSE", graph_json


# === Full Experiment ===
def run_full_experiment(K, model_name, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    preds_file = os.path.join(out_dir, f"predictions_K{K}_{model_name.replace(':','-')}.json")
    graphs_file = os.path.join(out_dir, f"graphs_K{K}_{model_name.replace(':','-')}.json")

    # Load dataset
    file_path = hf_hub_download("Yuqicheng/ArgRAG", "PubHealth.pkl", repo_type="dataset")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Load existing predictions and graphs
    results = json.load(open(preds_file, "r")) if os.path.exists(preds_file) else []
    graphs = json.load(open(graphs_file, "r")) if os.path.exists(graphs_file) else []

    completed_indices = {r["index"] for r in results}

    for idx, (claim, contexts, gt) in enumerate(zip(data["claims"], data["contexts"], data["answers"])):
        if idx in completed_indices:
            print(f"[{idx}] Already generated. Skipping...")
            continue

        print(f"[{idx}] Generating prediction and graph...")
        prediction, graph_json = run_arg_rag_prediction(claim, contexts, K, model_name)

        # Graph status
        graph_status = "generated" if graph_json["nodes"] is not None else "not_generated"

        graphs.append({
            "index": idx,
            "nodes": graph_json["nodes"],
            "edges": graph_json["edges"],
            "status": graph_status
        })

        results.append({
            "index": idx,
            "claim": claim,
            "prediction": prediction,
            "ground_truth": gt.strip().upper(),
            "model": model_name,
            "K": K,
            "graph_status": graph_status,
            "graph": {"nodes": graph_json["nodes"], "edges": graph_json["edges"]}
        })

        # Save immediately
        with open(preds_file, "w") as f:
            json.dump(results, f, indent=2)
        with open(graphs_file, "w") as f:
            json.dump(graphs, f, indent=2)

        print(f"[{idx}] Prediction saved: {prediction}, Graph status: {graph_status}")

    # Compute final metrics
    valid_results = [r for r in results if r["prediction"] in ["TRUE", "FALSE"]]
    y_true = [r["ground_truth"] for r in valid_results]
    y_pred = [r["prediction"] for r in valid_results]
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1 = f1_score(y_true, y_pred, pos_label="TRUE") if y_true else 0.0

    print(f"\nFinal Predictions saved to {preds_file}")
    print(f"Final Graphs saved to {graphs_file}")
    print(f"Final Accuracy: {acc:.3f}, F1: {f1:.3f}")

    return results, acc, f1, graphs


# === Plotting ===
def plot_accuracy(summary, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(summary)
    pivot = df.pivot(index="model", columns="K", values="accuracy")
    pivot.plot(kind="bar", figsize=(8, 5), title="ArgRAG Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend(title="K")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "argRAG_accuracy.png"))
    plt.close()


# === Main Loop ===
if __name__ == "__main__":
    Ks = [5]  # example
    MODELS = ["gpt-oss:20b"]  # example
    base_dir = "results"

    for K in Ks:
        for MODEL in MODELS:
            safe_model = MODEL.replace(":", "-")  # clean for filenames
            exp_dir = os.path.join(base_dir, f"{K}_{safe_model}")
            os.makedirs(exp_dir, exist_ok=True)

            print(f"\nRunning experiment for model={MODEL}, K={K}")
            results, acc, f1, graphs = run_full_experiment(K, MODEL, out_dir=exp_dir)

            # Prepare summary
            summary = [{"model": MODEL, "K": K, "accuracy": acc, "f1_score": f1}]

            # Save outputs
            preds_file = os.path.join(exp_dir, "predictions.json")
            graphs_file = os.path.join(exp_dir, "graphs.json")
            summary_file = os.path.join(exp_dir, "summary.json")

            with open(preds_file, "w") as f:
                json.dump(results, f, indent=2)
            with open(graphs_file, "w") as f:
                json.dump(graphs, f, indent=2)
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"Saved predictions → {preds_file}")
            print(f"Saved graphs → {graphs_file}")
            print(f"Saved summary → {summary_file}")
