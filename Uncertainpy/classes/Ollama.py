import subprocess
import json
import ollama


def ensure_ollama_model(model_name: str):
    """
    Ensure the Ollama model is available locally.
    If it's not, attempt to pull it.
    """
    try:
        # List installed models
        result = subprocess.run(["ollama", "list", "--json"], capture_output=True, text=True)
        if result.returncode == 0:
            try:
                installed = json.loads(result.stdout)
            except json.JSONDecodeError:
                installed = []

            model_names = [m["name"] for m in installed]
            if model_name in model_names:
                print(f"Model '{model_name}' is already present.")
                return

        print(f"Pulling model '{model_name}' ...")
        pull_result = subprocess.run(["ollama", "pull", model_name])
        if pull_result.returncode != 0:
            print(f"⚠️ Failed to pull model '{model_name}'")
    except Exception as e:
        print(f"⚠️ ensure_ollama_model error: {e}")


def run_ollama_inference(prompt: str, model: str) -> str:
    """
    Run inference using Ollama. 
    Tries both chat and generate APIs.
    Always returns a string (may be empty).
    """
    try:
        # Try chat API first
        res = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        if res and "message" in res and "content" in res["message"]:
            return res["message"]["content"].strip()

        # Fallback: try generate API
        res = ollama.generate(model=model, prompt=prompt)
        if res and "response" in res:
            return res["response"].strip()

        return ""  # nothing usable
    except Exception as e:
        print(f"⚠️ Ollama inference error: {e}")
        return ""
