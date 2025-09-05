from huggingface_hub import hf_hub_download
import pickle

# Download from Hugging Face Hub
file_path = hf_hub_download(
    repo_id="Yuqicheng/ArgRAG",  # your dataset repo
    filename="PubHealth.pkl",    # the exact filename
    repo_type="dataset"
)

# Load the pickle file
with open(file_path, "rb") as f:
    data = pickle.load(f)

# get claim, contexts and ground truth answer
claim = data["claims"][0]
contexts = data["contexts"][0]
gt_answer = data["answers"][0]


print(claim)
print(contexts)