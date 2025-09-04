import os
import json
import pickle
import re
import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, f1_score
import ollama
from classes.Ollama import ensure_ollama_model, run_ollama_inference
from classes.prompt import *
from uncertainpy.gradual import Argument, BAG
import networkx as nx
import uncertainpy.gradual as grad

# --------------------
# Helper: parse model answers
# --------------------
def extract_first_true_false(text):
    if not text:
        return None
    m = re.search(r'\b(true|false)\b', text, re.IGNORECASE)
    if m:
        return "TRUE" if m.group(1).lower() == "true" else "FALSE"
    return None

def extract_score_from_text(text):
    if not text:
        return None
    m = re.search(r'(\d{1,3}(?:\.\d+)?)', text)
    if m:
        val = float(m.group(1))
        if 0 <= val <= 100:
            return val
    return None

def parse_prediction(answer, prompt_type):
    if not answer:
        return "INVALID"
    if prompt_type == "binary":
        tf = extract_first_true_false(answer)
        return tf if tf else "INVALID"
    if prompt_type == "json_score":
        try:
            obj = json.loads(answer)
            if "score" in obj:
                score = float(obj["score"])
                return "TRUE" if score >= 50 else "FALSE"
        except:
            pass
        score = extract_score_from_text(answer)
        if score is not None:
            return "TRUE" if score >= 50 else "FALSE"
        return "INVALID"
    return "INVALID"

# --------------------
# ArgRAG Graph Functions
# --------------------
def ArgRAG_pred(ev2c_dict, ev2ev_dict, arg_dict, arg_model):
    evidence = ev2c_dict.get("support", []) + ev2c_dict.get("contradict", [])
    if not evidence:
        return "use parametric answer", None, {}

    G = nx.DiGraph()
    G.add_node("claim", type="claim", text=arg_dict["Claim"], strength=0.5)
    for ev in evidence:
        G.add_node(ev, type="evidence", text=arg_dict.get(ev, ""), strength=0.5)

    for sup in ev2c_dict.get("support", []):
        if sup in arg_dict:
            G.add_edge(sup, "claim", relation="support")
    for att in ev2c_dict.get("contradict", []):
        if att in arg_dict:
            G.add_edge(att, "claim", relation="attack")

    for rel_type in ["support", "contradict"]:
        for pair in ev2ev_dict.get(rel_type, []):
            if isinstance(pair, list) and len(pair) == 2 and all(k in arg_dict for k in pair):
                G.add_edge(pair[0], pair[1], relation="support" if rel_type=="support" else "attack")
                G.add_edge(pair[1], pair[0], relation="support" if rel_type=="support" else "attack")

    bag = BAG()
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

    # Ensure all nodes have 'type', 'strength', 'text'
    for n in G.nodes:
        G.nodes[n].setdefault("type", "unknown")
        G.nodes[n].setdefault("strength", 0.0)
        G.nodes[n].setdefault("text", arg_dict.get(n, ""))

    return ("true" if strengths["claim"] >= 0.5 else "false"), G, strengths

def run_arg_rag_prediction(claim, contexts, K, model_name="llama3.1"):
    contexts_to_use = contexts[:min(K, len(contexts))]
    arg_dict = {f"E{i+1}": ctx for i, ctx in enumerate(contexts_to_use)}
    arg_dict["Claim"] = claim

    # EV2C Prompt
    ev2c_prompt = EV2C_PROMPT.format(
        example=EV2C_JSON_EXAMPLE,
        claim=claim,
        evidence="".join(f"- E{i+1}: {ctx}\n" for i, ctx in enumerate(contexts_to_use))
    )
    ev2c_answer = run_ollama_inference(ev2c_prompt, model=model_name)
    ev2c_dict = safe_json_loads(ev2c_answer)

    if not (ev2c_dict.get("support") or ev2c_dict.get("contradict")):
        return "NO_EVIDENCE", {"index": None, "nodes": [], "edges": []}

    # EV2EV Prompt
    ev2ev_prompt = EV2EV_PROMPT.format(
        example=EV2EV_JSON_EXAMPLE,
        claim=claim,
        evidence="".join(f"- {k}: {arg_dict[k]}\n" for k in ev2c_dict.get("support", []) + ev2c_dict.get("contradict", []))
    )
    ev2ev_answer = run_ollama_inference(ev2ev_prompt, model=model_name)
    ev2ev_dict = safe_json_loads(ev2ev_answer)

    arg_model = grad.semantics.ContinuousDFQuADModel()
    pred, G, strengths = ArgRAG_pred(ev2c_dict, ev2ev_dict, arg_dict, arg_model)

    if G is None or len(G.nodes) == 0:
        return "NO_EVIDENCE", {"index": None, "nodes": [], "edges": []}

    graph_json = {
        "index": None,
        "claim": claim,
        "nodes": [
            {"id": n, "type": G.nodes[n].get("type", "unknown"),
             "strength": G.nodes[n].get("strength", 0.0),
             "text": G.nodes[n].get("text", arg_dict.get(n, ""))}
            for n in G.nodes
        ],
        "edges": [{"source": u, "target": v, "relation": d.get("relation", "unknown")} for u,v,d in G.edges(data=True)]
    }

    return "TRUE" if pred=="true" else "FALSE", graph_json

# --------------------
# Safe JSON Parsing
# --------------------
def safe_json_loads(json_str):
    default_dict = {"support": [], "contradict": [], "irrelevant": []}
    if not json_str or not isinstance(json_str, str):
        return default_dict
    json_str = re.sub(r'^```(?:json)?', '', json_str.strip(), flags=re.IGNORECASE)
    json_str = re.sub(r'```$', '', json_str)
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            for key in default_dict:
                data.setdefault(key, [])
            return data
    except:
        pass
    return default_dict

# --------------------
# Experiment Runner
# --------------------
def run_experiment(K, MODEL, prompt_name, prompt_template, prompt_type,
                   TEMP=0.3, MAX_TOKENS=64, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    model_safe = MODEL.replace(":", "-").replace("/", "-")
    predictions_file = os.path.join(save_dir, f"predictions_{prompt_name}_k{K}_{model_safe}.json")

    # Load dataset
    file_path = hf_hub_download("Yuqicheng/ArgRAG", "PubHealth.pkl", repo_type="dataset")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    ensure_ollama_model(MODEL)

    results = []
    for idx, (claim, contexts, gt) in enumerate(zip(data["claims"], data["contexts"], data["answers"])):
        cur_contexts = contexts[:min(K, len(contexts))]
        prediction, graph_json = run_arg_rag_prediction(claim, cur_contexts, K, model_name=MODEL)

        results.append({
            "index": idx,
            "claim": claim,
            "prediction": prediction,
            "ground_truth": gt.strip().upper(),
            "model": MODEL,
            "K": K,
            "prompt": prompt_name,
            "graph": graph_json
        })

        # Save intermediate results every 10 samples
        if idx % 10 == 0 and idx > 0:
            with open(predictions_file, "w") as f:
                json.dump(results, f, indent=2)

    # Save final predictions
    with open(predictions_file, "w") as f:
        json.dump(results, f, indent=2)

    # Accuracy and F1
    y_true = [r["ground_truth"] for r in results]
    y_pred = [r["prediction"] for r in results]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label="TRUE")
    print(f"{prompt_name} | {MODEL} | K={K} → Accuracy: {acc:.2%}, F1: {f1:.2%}")
    print(f"Predictions saved to {predictions_file}")

    return predictions_file, acc, f1, results

# --------------------
# Plotting
# --------------------
def plot_accuracy(summary, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(summary)

    for prompt in df["prompt"].unique():
        sub = df[df["prompt"] == prompt]

        pivot_acc = sub.pivot(index="model", columns="K", values="accuracy")
        pivot_acc.plot(kind="bar", figsize=(8,5), title=f"Accuracy for {prompt}")
        plt.ylabel("Accuracy")
        plt.ylim(0,1)
        plt.legend(title="K")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"accuracy_{prompt}.png"))
        plt.close()

        pivot_f1 = sub.pivot(index="model", columns="K", values="f1_score")
        pivot_f1.plot(kind="bar", figsize=(8,5), title=f"F1 Score for {prompt}")
        plt.ylabel("F1 Score")
        plt.ylim(0,1)
        plt.legend(title="K")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"f1_{prompt}.png"))
        plt.close()

# --------------------
# Main Loop
# --------------------
if __name__ == "__main__":
    Ks = [5, 10]
    MODELS = ["gemma3:270m", "gemma3:1b", "gemma3:4b", "gemma3:12b", "gemma3:27b"]

    TEMP = 0.0
    MAX_TOKENS = 16

    summary = []

    for K in Ks:
        for MODEL in MODELS:
            for prompt_name, (template, ptype) in FACTCHECK_PROMPTS.items():
                pred_file, acc, f1, detailed_results = run_experiment(
                    K, MODEL, prompt_name, template, ptype,
                    TEMP=TEMP, MAX_TOKENS=MAX_TOKENS,
                    save_dir="results"
                )
                summary.append({
                    "prompt": prompt_name,
                    "model": MODEL,
                    "K": K,
                    "accuracy": acc,
                    "f1_score": f1,
                    "pred_file": pred_file
                })

    # Save summary
    os.makedirs("results", exist_ok=True)
    summary_file = os.path.join("results", "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Generate plots
    plot_accuracy(summary)
    print("Plots saved in plots/ folder")
    print(f"Summary saved to {summary_file}")
