import os
import json
import pickle
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

# === Sample Claim and Evidence Texts ===
SAMPLE_CLAIM = "A mother revealed to her child in a letter after her death that she had just one eye because she had donated the other to him."

SAMPLE_EVIDENCES = {
    "E1": "Kastellorizo: crystal carafe with it and brought it to her future mother-in-law...",
    "E2": "Diane Schuur: Paulo, Rome, Palermo, Guanajuato, and multiple cities across the United States...",
    "E3": "Janis Babson: that when she died she wanted to donate her eyes to the Eye Bank...",
    "E4": "Portrait of the Artist's Mother at the Age of 63: After her death her grieving son wrote...",
    "E5": "The Eye (2002 film): donor. When they ask a village doctor about Ling and her family...",
    "E6": "Guddu: his dad will not even consider this. Things take a turn for the worse...",
    "E7": "Nicholas Hughes: literary and biographical writings about her death...",
    "E8": "Florida Hospital Wauchula: Mays were born within days of each other and were switched...",
    "E9": "One-Eye, Two-Eyes, and Three-Eyes: unkind than before. One day a Knight came riding along...",
    "E10": "Ruth Benedict: Because of this, the psychological effects on her childhood were profound..."
}

# === Sample Relationships Fully Embedded ===
PAIRWISE_PROMPT_WITH_TEXT_SAMPLE = """
Task:

You are given two textual items: Item A and Item B. 
Your job is to classify the relationship between them. 
Item A may support, contradict, or be unrelated to Item B.

Use the following sample as a reference:

Claim: {sample_claim}

Evidences and their texts:
{sample_evidences_texts}

Relationship between evidence and claim: 
support → E6, E8, E9
contradict → E3
irrelevant → E1, E2, E4, E5, E7, E10

Relationship between evidences: 
support → E6 → E8, E6 → E9
contradict → E3 → E6

Return exactly one label: "support", "contradict", or "unrelated".

Item A:
{item_a}

Item B:
{item_b}
"""

# === Pairwise Classification with Sample Guidance ===
def classify_pair(item_a_id, item_a_text, item_b_id, item_b_text, model_name="llama3.1"):
    sample_evidences_texts = "\n".join([f"{k}: {v}" for k, v in SAMPLE_EVIDENCES.items()])
    
    prompt = PAIRWISE_PROMPT_WITH_TEXT_SAMPLE.format(
        sample_claim=SAMPLE_CLAIM,
        sample_evidences_texts=sample_evidences_texts,
        item_a=item_a_text,
        item_b=item_b_text
    )
    
    response = run_ollama_inference(prompt, model=model_name).strip().lower()

    if "support" in response or "back" in response or "strengthen" in response:
        relation = "support"
    elif "contradict" in response or "attack" in response or "challenge" in response:
        relation = "contradict"
    else:
        relation = "unrelated"

    return (item_a_id, item_b_id, relation)

# === Argumentation Graph Construction ===
def ArgRAG_pred_divided(relations, arg_dict, arg_model):
    if not relations:
        return "use parametric answer", None, {}

    G = nx.DiGraph()
    for k, v in arg_dict.items():
        node_type = "claim" if k == "Claim" else "evidence"
        G.add_node(k, type=node_type, text=v)

    for src, tgt, rel in relations:
        if src in arg_dict and tgt in arg_dict:
            if rel in ["support", "contradict"]:
                G.add_edge(src, tgt, relation=rel)

    bag = grad.BAG()
    for n in G.nodes:
        bag.arguments[n] = Argument(n, 0.5)

    for u, v, d in G.edges(data=True):
        if d["relation"] == "support":
            bag.add_support(bag.arguments[u], bag.arguments[v])
        elif d["relation"] == "contradict":
            bag.add_attack(bag.arguments[u], bag.arguments[v])

    arg_model.BAG = bag
    arg_model.approximator = grad.algorithms.RK4(arg_model)
    arg_model.solve(delta=1e-2, epsilon=1e-4)

    strengths = {a.name: a.strength for a in bag.arguments.values()}
    nx.set_node_attributes(G, strengths, "strength")

    return ("true" if strengths.get("Claim", 0) >= 0.5 else "false"), G, strengths

# === Prediction Function ===
def run_arg_rag_prediction(claim, contexts, K, model_name="llama3.1", include_claim=False):
    contexts_to_use = contexts[:min(K, len(contexts))]
    arg_dict = {f"E{i+1}": ctx for i, ctx in enumerate(contexts_to_use)}
    if include_claim:
        arg_dict["Claim"] = claim

    arg_model = grad.semantics.ContinuousDFQuADModel()
    relations = []
    items = list(arg_dict.keys())

    # Pairwise classification for all pairs
    for i, a in enumerate(items):
        for j, b in enumerate(items):
            if i == j:
                continue
            rel = classify_pair(a, arg_dict[a], b, arg_dict[b], model_name)
            if rel[2] != "unrelated":
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
def run_full_experiment(K, model_name, out_dir="results", limit=None, include_claim=False):
    os.makedirs(out_dir, exist_ok=True)

    file_path = hf_hub_download("Yuqicheng/ArgRAG", "PubHealth.pkl", repo_type="dataset")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    total_examples = len(data["claims"])
    if limit is not None:
        limit = min(limit, total_examples)
    else:
        limit = total_examples

    results, graphs = [], []

    for idx, (claim, contexts, gt) in enumerate(zip(data["claims"], data["contexts"], data["answers"])):
        if idx >= limit:
            break

        print(f"[{idx}] Generating prediction and graph...")
        prediction, graph_json = run_arg_rag_prediction(claim, contexts, K, model_name, include_claim=include_claim)

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

        print(f"[{idx}] Prediction saved: {prediction}, Graph status: {graph_status}")

    valid_results = [r for r in results if r["prediction"] in ["TRUE", "FALSE"]]
    y_true = [r["ground_truth"] for r in valid_results]
    y_pred = [r["prediction"] for r in valid_results]
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1 = f1_score(y_true, y_pred, pos_label="TRUE") if y_true else 0.0

    print(f"\nFinal Accuracy: {acc:.3f}, F1: {f1:.3f}")

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
    Ks = [5]
    MODELS = ["gpt-oss:20b"]
    base_dir = "results"
    LIMIT = 3

    for K in Ks:
        for MODEL in MODELS:
            safe_model = MODEL.replace(":", "-")
            exp_dir = os.path.join(base_dir, f"{K}_{safe_model}")
            os.makedirs(exp_dir, exist_ok=True)

            print(f"\nRunning experiment for model={MODEL}, K={K}, limit={LIMIT}")

            results, acc, f1, graphs = run_full_experiment(K, MODEL, out_dir=exp_dir, limit=LIMIT, include_claim=False)

            summary = [{"model": MODEL, "K": K, "accuracy": acc, "f1_score": f1}]
            with open(os.path.join(exp_dir, "predictions.json"), "w") as f:
                json.dump(results, f, indent=2)
            with open(os.path.join(exp_dir, "graphs.json"), "w") as f:
                json.dump(graphs, f, indent=2)
            with open(os.path.join(exp_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

            print(f"Saved predictions → {os.path.join(exp_dir, 'predictions.json')}")
            print(f"Saved graphs → {os.path.join(exp_dir, 'graphs.json')}")
            print(f"Saved summary → {os.path.join(exp_dir, 'summary.json')}")
