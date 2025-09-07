import os
import json
import pickle
import re
import networkx as nx
from huggingface_hub import hf_hub_download
import ollama
import src.uncertainpy.gradual as grad
from src.uncertainpy.gradual import Argument, BAG

# === Prompts & Examples ===
EV2C_PROMPT = """
Task: Given a claim and multiple pieces of evidence, classify each evidence as "support", "contradict", or "irrelevant" to the claim.
Output Format:
Return a JSON object with keys: "support", "contradict", "irrelevant".
Example: {example}
Claim: {claim}
Evidence:
{evidence}
"""
EV2C_JSON_EXAMPLE = '{"support": ["E1"], "contradict": ["E3"], "irrelevant": ["E2"]}'

EV2EV_PROMPT = """
Task: Given a claim and multiple pieces of evidence, analyze relationships between evidence: "support" or "contradict".
Output Format: JSON object with keys: "support", "contradict".
Example: {example}
Claim: {claim}
Evidence:
{evidence}
"""
EV2EV_JSON_EXAMPLE = '{"support": [["E1","E2"]], "contradict": [["E2","E3"]]}'

# === Ollama inference ===
def run_ollama_inference(prompt, model="llama3.1"):
    try:
        resp = ollama.chat(model=model, messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])
        if not resp or "message" not in resp or "content" not in resp["message"]:
            return "{}"
        return resp["message"]["content"]
    except:
        return "{}"

# === Safe JSON parser ===
def safe_json_loads(json_str):
    default = {"support": [], "contradict": [], "irrelevant": []}
    if not json_str: return default
    json_str = re.sub(r'^```(?:json)?', '', json_str.strip(), flags=re.IGNORECASE)
    json_str = re.sub(r'```$', '', json_str).strip()
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            for k in default: data.setdefault(k, [])
            return data
    except: pass
    return default

# === ArgRAG graph computation ===
def ArgRAG_pred(ev2c, ev2ev, arg_dict, arg_model):
    evidence = ev2c.get("support", []) + ev2c.get("contradict", [])
    if not evidence:
        return "use parametric answer", None, {}
    G = nx.DiGraph()
    G.add_node("claim", type="claim", text=arg_dict["Claim"], strength=0.5)
    for ev in evidence:
        G.add_node(ev, type="evidence", text=arg_dict.get(ev, ""), strength=0.5)
    for sup in ev2c.get("support", []):
        if sup in arg_dict: G.add_edge(sup, "claim", relation="support")
    for att in ev2c.get("contradict", []):
        if att in arg_dict: G.add_edge(att, "claim", relation="attack")
    for rel_type in ["support", "contradict"]:
        for pair in ev2ev.get(rel_type, []):
            if isinstance(pair, list) and len(pair)==2 and all(k in arg_dict for k in pair):
                G.add_edge(pair[0], pair[1], relation="support" if rel_type=="support" else "attack")
                G.add_edge(pair[1], pair[0], relation="support" if rel_type=="support" else "attack")
    bag = BAG()
    for n in G.nodes: bag.arguments[n] = Argument(n, 0.5)
    for u,v,d in G.edges(data=True):
        if d["relation"]=="support": bag.add_support(bag.arguments[u], bag.arguments[v])
        else: bag.add_attack(bag.arguments[u], bag.arguments[v])
    arg_model.BAG = bag
    arg_model.approximator = grad.algorithms.RK4(arg_model)
    arg_model.solve(delta=1e-2, epsilon=1e-4)
    strengths = {a.name: a.strength for a in bag.arguments.values()}
    nx.set_node_attributes(G, strengths, "strength")
    return ("true" if strengths["claim"]>=0.5 else "false"), G, strengths

# === Single prediction ===
def run_arg_rag_prediction(claim, contexts, K, model_name="llama3.1"):
    contexts = contexts[:min(K,len(contexts))]
    arg_dict = {f"E{i+1}": ctx for i, ctx in enumerate(contexts)}
    arg_dict["Claim"] = claim
    ev2c_prompt = EV2C_PROMPT.format(example=EV2C_JSON_EXAMPLE, claim=claim,
                                     evidence="".join(f"- E{i+1}: {ctx}\n" for i,ctx in enumerate(contexts)))
    ev2c_answer = run_ollama_inference(ev2c_prompt, model=model_name)
    ev2c_dict = safe_json_loads(ev2c_answer)
    if not (ev2c_dict.get("support") or ev2c_dict.get("contradict")):
        return "NO_EVIDENCE", {"index": None, "nodes": [], "edges": []}
    ev2ev_prompt = EV2EV_PROMPT.format(example=EV2EV_JSON_EXAMPLE, claim=claim,
                                       evidence="".join(f"- {k}: {arg_dict[k]}\n" for k in ev2c_dict.get("support",[])+ev2c_dict.get("contradict",[])))
    ev2ev_answer = run_ollama_inference(ev2ev_prompt, model=model_name)
    ev2ev_dict = safe_json_loads(ev2ev_answer)
    arg_model = grad.semantics.ContinuousDFQuADModel()
    pred, G, strengths = ArgRAG_pred(ev2c_dict, ev2ev_dict, arg_dict, arg_model)
    if G is None or len(G.nodes)==0:
        return "NO_EVIDENCE", {"index": None, "nodes": [], "edges": []}
    graph_json = {
        "index": None,
        "claim": claim,
        "nodes":[{"id":n,"type":G.nodes[n].get("type","unknown"),
                  "strength":G.nodes[n].get("strength",0.0),
                  "text":G.nodes[n].get("text",arg_dict.get(n,""))} for n in G.nodes],
        "edges":[{"source":u,"target":v,"relation":d.get("relation","unknown")} for u,v,d in G.edges(data=True)]
    }
    return "TRUE" if pred=="true" else "FALSE", graph_json

# === Master summary updater ===
def update_master_summary(new_summary, master_summary_file="results/master_summary.json"):
    os.makedirs(os.path.dirname(master_summary_file), exist_ok=True)
    if os.path.exists(master_summary_file):
        with open(master_summary_file,"r") as f: master = json.load(f)
    else: master = []
    existing_keys = {(e.get("model"), e.get("K")) for e in master}
    for e in new_summary:
        key = (e.get("model"), e.get("K"))
        if key not in existing_keys:
            master.append(e)
            existing_keys.add(key)
    with open(master_summary_file,"w") as f: json.dump(master,f,indent=2)
    print(f"Master summary updated: {master_summary_file}")

# === Full Experiment Runner (with resume) ===
def run_full_experiment(K, model_name="llama3.1", out_dir="results"):
    os.makedirs(out_dir,exist_ok=True)
    preds_file = os.path.join(out_dir,f"predictions_K{K}_{model_name.replace(':','-')}.json")
    graphs_file = os.path.join(out_dir,f"graphs_K{K}_{model_name.replace(':','-')}.json")

    # Load dataset
    file_path = hf_hub_download("Yuqicheng/ArgRAG","PubHealth.pkl",repo_type="dataset")
    with open(file_path,"rb") as f: data = pickle.load(f)

    # Load existing results if any
    existing_results = {}
    if os.path.exists(preds_file):
        with open(preds_file, "r") as f:
            existing_results = {r["index"]: r for r in json.load(f)}

    results, graphs = [], []
    for idx, (claim, contexts, gt) in enumerate(zip(data["claims"],data["contexts"],data["answers"])):
        if idx in existing_results:
            print(f"[{idx}] Already processed, skipping...")
            results.append(existing_results[idx])
            graphs.append({"index": idx,
                           "nodes": existing_results[idx]["graph"].get("nodes"),
                           "edges": existing_results[idx]["graph"].get("edges")})
            continue

        print(f"[{idx}] Generating...")
        pred, graph_json = run_arg_rag_prediction(claim, contexts, K, model_name)
        results.append({"index": idx, "claim": claim, "prediction": pred, "model": model_name, "K": K, "graph": graph_json})
        graphs.append({"index": idx, "nodes": graph_json.get("nodes"), "edges": graph_json.get("edges")})

        # Save every 10 iterations
        if idx % 10 == 0 and idx > 0:
            with open(preds_file, "w") as f: json.dump(results, f, indent=2)
            with open(graphs_file, "w") as f: json.dump(graphs, f, indent=2)

    # Final save
    with open(preds_file, "w") as f: json.dump(results, f, indent=2)
    with open(graphs_file, "w") as f: json.dump(graphs, f, indent=2)

    # Save summary (without metrics)
    summary = [{"model": model_name, "K": K}]
    summary_file = os.path.join(out_dir, "summary.json")
    with open(summary_file, "w") as f: json.dump(summary, f, indent=2)
    update_master_summary(summary, master_summary_file="results/master_summary.json")
    return results, graphs

# === Main Loop ===
if __name__=="__main__":
    Ks = [5,10]
    MODELS = ["gemma3:270m","gemma3:1b","gemma3:4b","gemma3:12b"]
    base_dir = "results"
    for K in Ks:
        for MODEL in MODELS:
            run_full_experiment(K, MODEL, out_dir=base_dir)
