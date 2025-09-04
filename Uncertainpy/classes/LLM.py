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



# =============== FUNZIONI LLM ==================
def run_ollama_inference(prompt, model, temperature, max_tokens):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a reasoning assistant. Always return valid JSON."},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": temperature, "num_predict": max_tokens}
    )
    return response["message"]["content"]

def safe_json_loads(json_str):
    json_str = json_str.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        fixed_str = json_str
        fixed_str = re.sub(r'\((\w+),\s*(\w+)\)', r'["\1", "\2"]', fixed_str)
        fixed_str = re.sub(r'\[(\w+),\s*(\w+)\]', r'["\1", "\2"]', fixed_str)
        for key in ["support", "contradict", "irrelevant"]:
            pattern = rf'"{key}":\s*\[(.*?)\]'
            match = re.search(pattern, fixed_str)
            if match:
                elements = match.group(1).split(',')
                quoted = [f'"{e.strip()}"' for e in elements if e.strip()]
                replacement = f'"{key}": [{", ".join(quoted)}]'
                fixed_str = re.sub(pattern, replacement, fixed_str)
        try:
            return json.loads(fixed_str)
        except:
            return {}
# ===============================================