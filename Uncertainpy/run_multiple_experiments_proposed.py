import os
import json
import pickle
import random
import networkx as nx
from huggingface_hub import hf_hub_download
import ollama
import src.uncertainpy.gradual as grad
from src.uncertainpy.gradual import Argument, BAG
from sklearn.metrics import accuracy_score, f1_score

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

# === Classification Functions (LLM-only) ===
def classify_claim_evidence(claim, ev_id, ev_text, model_name="llama3.1"):
    prompt = f"""
Task:
Given a claim and one piece of evidence, classify the evidence in relation to the claim.
Choose exactly one: "support", "attack", "not_related".

Output Format:
Return one label: "support", "attack", or "not_related".

Claim:
{claim}

Evidence:
{ev_text}
"""
    response = run_ollama_inference(prompt, model=model_name).strip().lower()
    if "support" in response:
        rel = "support"
    elif "attack" in response:
        rel = "attack"
    else:
        rel = "not_related"
    return (ev_id, "Claim", rel)

def classify_evidence_evidence(ev1_id, ev1_text, ev2_id, ev2_text, model_name="llama3.1"):
    prompt = f"""
Task:
Given two pieces of evidence, classify the relation of the first to the second.
Choose exactly one: "support", "attack", "not_related".

Output Format:
Return one label: "support", "attack", or "not_related".

Evidence A:
{ev1_text}

Evidence B:
{ev2_text}
"""
    response = run_ollama_inference(prompt, model=model_name).strip().lower()
    if "support" in response:
        rel = "support"
    elif "attack" in response:
        rel = "attack"
    else:
        rel = "not_related"
    return (ev1_id, ev2_id, rel)

# === Argumentative Graph with BAG Calculation ===
def ArgRAG_pred_with_calculus(relations, arg_dict):
    if not relations:
        return "use parametric answer", None, {}

    G = nx.DiGraph()
    for k, v in arg_dict.items():
        node_type = "claim" if k == "Claim" else "evidence"
        G.add_node(k, type=node_type, text=v, strength=0.5)

    for src, tgt, rel in relations:
        if src in arg_dict and tgt in arg_dict:
            G.add_edge(src, tgt, relation=rel)

    bag = BAG()
    for n in G.nodes:
        bag.arguments[n] = Argument(n, initial_weight=0.5)

    for u, v, d in G.edges(data=True):
        if d["relation"] == "support":
            bag.add_support(bag.arguments[u], bag.arguments[v])
        elif d["relation"] == "attack":
            bag.add_attack(bag.arguments[u], bag.arguments[v])

    arg_model = grad.semantics.ContinuousDFQuADModel()
    arg_model.BAG = bag
    arg_model.approximator = grad.algorithms.RK4(arg_model)
    arg_model.solve(delta=1e-2, epsilon=1e-4)

    strengths = {a.name: a.strength for a in bag.arguments.values()}
    nx.set_node_attributes(G, strengths, "strength")

    return ("true" if strengths.get("Claim", 0) >= 0.5 else "false"), G, strengths

# === Prediction Function with LLM-extracted Graph ===
def run_arg_rag_prediction(claim, contexts, K, model_name="llama3.1"):
    contexts_to_use = contexts[:min(K, len(contexts))]
    arg_dict = {f"E{i+1}": ctx for i, ctx in enumerate(contexts_to_use)}
    arg_dict["Claim"] = claim

    evidences = [f"E{i+1}" for i in range(len(contexts_to_use))]
    relations = []
    llm_predictions = []

    # Evidence → Claim edges
    for ev in evidences:
        src, tgt, rel = classify_claim_evidence(claim, ev, arg_dict[ev], model_name)
        llm_predictions.append((src, tgt, rel))
        relations.append((src, tgt, rel))

    # Evidence → Evidence edges
    for i, e1 in enumerate(evidences):
        for j, e2 in enumerate(evidences):
            if i == j:
                continue
            src, tgt, rel = classify_evidence_evidence(e1, arg_dict[e1], e2, arg_dict[e2], model_name)
            relations.append((src, tgt, rel))
            llm_predictions.append((src, tgt, rel))

    # Print all LLM predictions
    print("\n=== All LLM Predicted Relations ===")
    for src, tgt, rel in llm_predictions:
        print(f"{src} → {tgt}: {rel}")

    pred, G, strengths = ArgRAG_pred_with_calculus(relations, arg_dict)

    if G is None or len(G.nodes) == 0:
        return "NO_EVIDENCE", {"nodes": None, "edges": None}

    graph_json = {
        "nodes": [{"id": n,
                   "type": G.nodes[n].get("type", "unknown"),
                   "strength": G.nodes[n].get("strength", 0.0),
                   "text": G.nodes[n].get("text", arg_dict.get(n, ""))}
                  for n in G.nodes],
        "edges": [{"source": u, "target": v, "relation": d.get("relation", "unknown")} for u, v, d in G.edges(data=True)]
    }
    return "TRUE" if pred == "true" else "FALSE", graph_json

# === Full Experiment Runner with cumulative accuracy ===
def run_full_experiment(K, model_name, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    preds_file = os.path.join(out_dir, f"predictions_K{K}_{model_name.replace(':','-')}.json")
    graphs_file = os.path.join(out_dir, f"graphs_K{K}_{model_name.replace(':','-')}.json")

    file_path = hf_hub_download("Yuqicheng/ArgRAG", "PubHealth.pkl", repo_type="dataset")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    results = json.load(open(preds_file, "r")) if os.path.exists(preds_file) else []
    graphs = json.load(open(graphs_file, "r")) if os.path.exists(graphs_file) else []

    completed_indices = {r["index"] for r in results}
    cumulative_correct = 0
    cumulative_total = 0

    for idx, (claim, contexts, gt) in enumerate(zip(data["claims"], data["contexts"], data["answers"])):
        if idx in completed_indices:
            # Update cumulative counters based on previous results
            prev_pred = next(r["prediction"] for r in results if r["index"] == idx)
            cumulative_total += 1
            if prev_pred.upper() == gt.strip().upper():
                cumulative_correct += 1
            continue

        print(f"[{idx}] Generating prediction...")
        prediction, graph_json = run_arg_rag_prediction(claim, contexts, K, model_name)
        graph_status = "generated" if graph_json["nodes"] is not None else "not_generated"

        # Update cumulative counters
        cumulative_total += 1
        if prediction.upper() == gt.strip().upper():
            cumulative_correct += 1
        cumulative_accuracy = cumulative_correct / cumulative_total
        print(f"[{idx}] Prediction: {prediction}, Ground Truth: {gt.strip().upper()}, Cumulative Accuracy: {cumulative_accuracy:.3f}")

        graphs.append({"index": idx, "nodes": graph_json["nodes"], "edges": graph_json["edges"],
                       "status": graph_status, "cumulative_accuracy": cumulative_accuracy})
        results.append({
            "index": idx,
            "claim": claim,
            "prediction": prediction,
            "ground_truth": gt.strip().upper(),
            "model": model_name,
            "K": K,
            "graph_status": graph_status,
            "cumulative_accuracy": cumulative_accuracy,
            "graph": {"nodes": graph_json["nodes"], "edges": graph_json["edges"]}
        })

        if idx % 5 == 0:
            with open(preds_file, "w") as f: json.dump(results, f, indent=2)
            with open(graphs_file, "w") as f: json.dump(graphs, f, indent=2)

    # Overall metrics
    valid_results = [r for r in results if r["prediction"] in ["TRUE", "FALSE"]]
    y_true = [r["ground_truth"] for r in valid_results]
    y_pred = [r["prediction"] for r in valid_results]
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1 = f1_score(y_true, y_pred, pos_label="TRUE") if y_true else 0.0
    print(f"Final Accuracy: {acc:.3f}, F1: {f1:.3f}")

    return results, acc, f1, graphs

# === Demo Function ===
def demo_prompts(model_name="llama3.1"):
    claim = "Eating an apple a day reduces the risk of heart disease."
    examples = [
        "A recent clinical trial found no improvement in heart health from daily apple consumption.",
        "Some studies show apples have a minimal effect on heart disease risk.",
        "A 10-year study found daily apple consumption was associated with a 20% lower risk of cardiovascular disease.",
        "Apples are rich in dietary fiber and antioxidants, which protect the heart.",
        "Apple trees require cold winters to produce fruit."
    ]

    print("\n=== Demo Classification with Strengths ===")
    prediction, graph_json = run_arg_rag_prediction(claim, examples, K=5, model_name=model_name)
    print(f"Claim Prediction: {prediction}")
    print("Graph Nodes and Strengths:")
    for node in graph_json["nodes"]:
        print(f"{node['id']}: strength={node.get('strength',0):.2f} → {node['text']}")

# === Main Loop ===
if __name__ == "__main__":
    random.seed(42)

    # Demo
    demo_prompts("gpt-oss:20b")

    # Full experiment
    Ks = [5, 10]
    MODELS = ["gpt-oss:20b"]
    base_dir = "results"

    for K in Ks:
        for MODEL in MODELS:
            exp_dir = os.path.join(base_dir, f"{K}_{MODEL.replace(':','-')}")
            os.makedirs(exp_dir, exist_ok=True)
            print(f"\nRunning experiment for model={MODEL}, K={K}")
            results, acc, f1, graphs = run_full_experiment(K, MODEL, out_dir=exp_dir)
