import os
import argparse
from huggingface_hub import hf_hub_download, snapshot_download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def run_init(hf_token, model_repo, model_file, models_dir, dataset_repo, data_dir):
    """Downloads the pretrained model and dataset from Hugging Face."""
    print(f"--- Starting Data Acquisition ---")
    
    # 1. Download Pretrained Model
    os.makedirs(models_dir, exist_ok=True)
    target_model_path = os.path.join(models_dir, model_file)
    
    if os.path.exists(target_model_path):
        print(f"Model already exists at {target_model_path}")
    else:
        print(f"Downloading model from {model_repo}...")
        try:
            hf_hub_download(
                repo_id=model_repo,
                filename=model_file,
                local_dir=models_dir,
                token=hf_token
            )
            print(f"Model successfully saved in {models_dir}/")
        except Exception as e:
            print(f"Error downloading model: {e}")

    # 2. Download Dataset
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading dataset from {dataset_repo}...")
    try:
        # snapshot_download retrieves the entire repository structure
        snapshot_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            local_dir=data_dir,
            token=hf_token,
            max_workers=32
        )
        print(f"Dataset successfully saved in {data_dir}/")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

    print(f"\n--- Initialization complete ---")

if __name__ == "__main__":
    # --- CLI Arguments ---
    parser = argparse.ArgumentParser(description="Initialize project environment")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Access Token")
    args = parser.parse_args()

    # --- Configuration Parameters ---
    CONFIG = {
        "hf_token": args.token,
        "model_repo": "facebook/VGGT-1B",
        "model_file": "model.pt",
        "models_dir": "models",
        "dataset_repo": "Uniiii/eth3d_v2",
        "data_dir": os.path.join("data", "eth3D")
    }

    # --- Execute ---
    run_init(**CONFIG)