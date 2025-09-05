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

# === Prompts & Examples ===
EV2C_PROMPT = """
Task: Given a claim and multiple pieces of evidence, classify each evidence as "support", "contradict", or "irrelevant" to the claim.
Classify each evidence as either supporting, contradicting, or irrelevant to the claim.
Instructions:
- Support: Evidence that backs the claim.
- Contradict: Evidence that counters or limits the claim.
- Irrelevant: Evidence unrelated to the claim.
Output Format:
Return a single JSON object with three keys: "support", "contradict", and "irrelevant", each mapping to a list of evidence items.
Example format: {example}
Claim:
{claim}
Evidence:
{evidence}
You must always PROVIDE ONLY A SINGLE JSON without any additional explanation or commentary.
**Do not** include markdown formatting (such as triple backticks or `json` tags) in the output.
"""

EV2C_JSON_EXAMPLE = '{"support": ["E1"], "contradict": ["E3", "E4"], "irrelevant": ["E2", "E5"]}'

EV2EV_PROMPT = """
Task: Given a claim and multiple pieces of evidence, analyze the relationships between evidence with respect to the claim.
Instructions:
- Support: Two evidence items that reinforce each other regarding the claim.
- Contradict: Two evidence items that conflict with each other regarding the claim.
Output Format:
Return a single JSON object with two keys: "support" and "contradict", each mapping to a list of evidence pairs.
Example format: {example}
Claim:
{claim}
Evidence:
{evidence}
You must always PROVIDE ONLY A SINGLE JSON without any additional explanation or commentary.
**Do not** include markdown formatting (such as triple backticks or `json` tags) in the output.
"""

EV2C_JSON_EXAMPLE = '{"support": ["E1"], "contradict": ["E3"], "irrelevant": ["E2"]}'
EV2EV_JSON_EXAMPLE = '{"support": [["E1","E2"]], "contradict": [["E2","E3"]]}'

# === Ollama Inference ===
def run_ollama_inference(prompt, model="llama3.1", temperature=0.0, max_tokens=512):
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

# === Safe JSON Parsing ===
def safe_json_loads(json_str):
    default_dict = {"support": [], "contradict": [], "irrelevant": []}
    if not json_str or not isinstance(json_str, str):
        return default_dict
    json_str = re.sub(r'^```(?:json)?', '', json_str.strip(), flags=re.IGNORECASE)
    json_str = re.sub(r'```$', '', json_str)
    json_str = json_str.strip()
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            for key in default_dict:
                data.setdefault(key, [])
            return data
    except:
        pass
    return default_dict

# === Argumentation Graph ===
def ArgRAG_pred(ev2c_dict, ev2ev_dict, arg_dict, arg_model):
    evidence = ev2c_dict.get("support", []) + ev2c_dict.get("contradict", [])
    if not evidence:
        return "use parametric answer", None, {}

    G = nx.DiGraph()
    G.add_node("claim", type="claim", text=arg_dict["Claim"])

    for ev in evidence:
        G.add_node(ev, type="evidence", text=arg_dict.get(ev, ""))

    for sup in ev2c_dict.get("support", []):
        if sup in arg_dict:
            G.add_edge(sup, "claim", relation="support")
    for att in ev2c_dict.get("contradict", []):
        if att in arg_dict:
            G.add_edge(att, "claim", relation="attack")

    for rel_type in ["support", "contradict"]:
        for pair in ev2ev_dict.get(rel_type, []):
            if isinstance(pair, list) and len(pair) == 2 and all(k in arg_dict for k in pair):
                G.add_edge(pair[0], pair[1], relation=rel_type if rel_type == "support" else "attack")
                G.add_edge(pair[1], pair[0], relation=rel_type if rel_type == "support" else "attack")

    bag = grad.BAG()
    for n in G.nodes:
        bag.arguments[n] = Argument(n, 0.5)

    for u, v, d in G.edges(data=True):
        if d["relation"] == "support":
            bag.add_support(bag.arguments[u], bag.arguments[v])
        else:
            bag.add_attack(bag.arguments[u], bag.arguments[v])

    arg_model.BAG = bag
    arg_model.approximator = grad.algorithms.RK4(arg_model)
    arg_model.solve(delta=1e-2, epsilon=1e-4)

    strengths = {a.name: a.strength for a in bag.arguments.values()}
    nx.set_node_attributes(G, strengths, "strength")

    return ("true" if strengths["claim"] >= 0.5 else "false"), G, strengths

# === Single Attempt Prediction ===
def run_arg_rag_prediction(claim, contexts, K, model_name="llama3.1"):
    contexts_to_use = contexts[:min(K, len(contexts))]
    arg_dict = {f"E{i+1}": ctx for i, ctx in enumerate(contexts_to_use)}
    arg_dict["Claim"] = claim

    # EV2C
    ev2c_prompt = EV2C_PROMPT.format(
        example=EV2C_JSON_EXAMPLE,
        claim=claim,
        evidence="".join(f"- E{i+1}: {ctx}\n" for i, ctx in enumerate(contexts_to_use))
    )
    ev2c_answer = run_ollama_inference(ev2c_prompt, model=model_name)
    ev2c_dict = safe_json_loads(ev2c_answer)

    if not (ev2c_dict.get("support") or ev2c_dict.get("contradict")):
        return "NO_EVIDENCE", {"nodes": None, "edges": None, "raw_llm": {"ev2c": ev2c_answer, "ev2ev": None}}

    # EV2EV
    ev2ev_prompt = EV2EV_PROMPT.format(
        example=EV2EV_JSON_EXAMPLE,
        claim=claim,
        evidence="".join(f"- {k}: {arg_dict[k]}\n" for k in ev2c_dict["support"] + ev2c_dict["contradict"])
    )
    ev2ev_answer = run_ollama_inference(ev2ev_prompt, model=model_name)
    ev2ev_dict = safe_json_loads(ev2ev_answer)

    # Build graph
    arg_model = grad.semantics.ContinuousDFQuADModel()
    pred, G, strengths = ArgRAG_pred(ev2c_dict, ev2ev_dict, arg_dict, arg_model)

    graph_json = {
        "nodes": None,
        "edges": None,
        "raw_llm": {"ev2c": ev2c_answer, "ev2ev": ev2ev_answer}
    }

    if G is not None and len(G.nodes) > 0:
        graph_json["nodes"] = [
            {
                "id": n,
                "type": G.nodes[n].get("type", "unknown"),
                "strength": G.nodes[n].get("strength", 0.5),
                "text": arg_dict.get(n, "")
            }
            for n in G.nodes
        ]
        graph_json["edges"] = [{"source": u, "target": v, "relation": d.get("relation", "unknown")} for u, v, d in G.edges(data=True)]

    return ("TRUE" if pred == "true" else "FALSE"), graph_json

# === Full Experiment ===
def run_full_experiment(K, model_name="llama3.1", out_dir="results"):
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

        # Determine graph status
        graph_status = "wrongly_generated" if graph_json["nodes"] is None else "generated"

        # Save graphs separately
        graphs.append({
            "index": idx,
            "nodes": graph_json.get("nodes"),
            "edges": graph_json.get("edges"),
            "status": graph_status,
            "raw_llm": graph_json.get("raw_llm")
        })

        # Save predictions
        results.append({
            "index": idx,
            "claim": claim,
            "prediction": prediction,
            "ground_truth": gt.strip().upper(),
            "model": model_name,
            "K": K,
            "graph_status": graph_status,
            "graph": {"nodes": graph_json.get("nodes"), "edges": graph_json.get("edges")},
            "raw_llm": graph_json.get("raw_llm")
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
    Ks = [10]  # example
    MODELS = ["gpt-oss:20b"]  # example
    base_dir = "results"
    preds_dir = os.path.join(base_dir, "predictions")
    summary_dir = os.path.join(base_dir, "summaries")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    summary = []
    all_predictions = {}
    summary_file = os.path.join(summary_dir, "summary.json")
    all_preds_file = os.path.join(preds_dir, "all_predictions.json")

    for K in Ks:
        for MODEL in MODELS:
            print(f"\nRunning experiment for model={MODEL}, K={K}")
            results, acc, f1, graphs = run_full_experiment(K, MODEL, out_dir=preds_dir)

            summary.append({"model": MODEL, "K": K, "accuracy": acc, "f1_score": f1})
            all_predictions[f"{MODEL}_K{K}"] = results

            # Save summary & all predictions incrementally
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            with open(all_preds_file, "w") as f:
                json.dump(all_predictions, f, indent=2)
            print(f"Updated summary saved to {summary_file}")
            print(f"Updated all predictions saved to {all_preds_file}")

    plot_accuracy(summary, output_dir=plots_dir)
    print(f"Plots saved in {plots_dir}/ folder")
