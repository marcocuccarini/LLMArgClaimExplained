import os
import sys
import json
import pickle
import re
import networkx as nx
import pandas as pd
from tqdm import tqdm

import ollama
from huggingface_hub import hf_hub_download
import uncertainpy.gradual as grad
from uncertainpy.gradual.Argument import Argument
from sklearn.metrics import classification_report, f1_score, accuracy_score




# =============== ARGUMENTATION =================
def ArgRAG_pred(ev2c_dict, ev2ev_dict, arg_dict, arg_model):
    evidence = ev2c_dict.get("support", []) + ev2c_dict.get("contradict", [])
    if len(evidence) == 0:
        return "use parametric answer", None

    G = nx.DiGraph()
    G.add_node("claim", type="claim", text=arg_dict["Claim"])
    for ev in evidence:
        if ev in arg_dict:
            G.add_node(ev, type="evidence", text=arg_dict[ev])

    for sup in ev2c_dict.get("support", []):
        if sup in G.nodes: G.add_edge(sup, "claim", relation="support")
    for att in ev2c_dict.get("contradict", []):
        if att in G.nodes: G.add_edge(att, "claim", relation="attack")

    for ev1, ev2 in ev2ev_dict.get("support", []):
        if ev1 in G.nodes and ev2 in G.nodes:
            G.add_edge(ev1, ev2, relation="support")
            G.add_edge(ev2, ev1, relation="support")
    for ev1, ev2 in ev2ev_dict.get("contradict", []):
        if ev1 in G.nodes and ev2 in G.nodes:
            G.add_edge(ev1, ev2, relation="attack")
            G.add_edge(ev2, ev1, relation="attack")

    bag = grad.BAG()
    for id, meta in G.nodes(data=True):
        from uncertainpy.gradual.Argument import Argument
        argument = Argument(id, 0.5)
        bag.arguments[id] = argument
    for u, v, d in G.edges(data=True):
        if d["relation"] == "support":
            bag.add_support(bag.arguments[u], bag.arguments[v])
        elif d["relation"] == "attack":
            bag.add_attack(bag.arguments[u], bag.arguments[v])

    arg_model.BAG = bag
    arg_model.approximator = grad.algorithms.RK4(arg_model)
    arg_model.solve(delta=1e-2, epsilon=1e-4, verbose=False, generate_plot=False)

    arg_strengths = {x.name: x.strength for x in arg_model.BAG.arguments.values()}
    G.nodes["claim"]["strength"] = arg_strengths.get("claim", 0.0)
    for ev in evidence:
        if ev in arg_strengths:
            G.nodes[ev]["strength"] = arg_strengths[ev]

    return ("true" if arg_strengths.get("claim", 0.0) >= 0.5 else "false"), G
# ===============================================