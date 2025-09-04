import os
import sys
import json
import pickle
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
import subprocess
import ollama


def run_ollama_inference(prompt, model="gemma3:1b", temperature=0.3, max_tokens=64):
    """
    Send a prompt to the Ollama model and return the response text.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"Error in Ollama inference: {e}")
        return ""


def ensure_ollama_model(model_name):
    """
    Checks if the Ollama model is present locally. If not, downloads it.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        models = [line.split()[0] for line in result.stdout.strip().split('\n')[1:] if line]
        if model_name not in models:
            print(f"Model '{model_name}' not found locally. Downloading...")
            pull_result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True
            )
            if pull_result.returncode != 0:
                print(f"Failed to download model '{model_name}'. Error:\n{pull_result.stderr}")
                sys.exit(1)
            print(f"Model '{model_name}' downloaded successfully.")
        else:
            print(f"Model '{model_name}' is already present.")
    except FileNotFoundError:
        print("Ollama CLI not found. Please install Ollama and ensure it is in your PATH.")
        sys.exit(1)
    except Exception as e:
        print(f"Error checking/downloading Ollama model: {e}")
        sys.exit(1)



