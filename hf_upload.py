from huggingface_hub import HfApi
import argparse

def upload_model(repo_name):
    api = HfApi()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_folder(
        folder_path="./models/llama3b_finetuned",
        repo_id=repo_name
    )
    print(f"Model uploaded to Hugging Face: econstellation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_name", type=str, default="vishalzaiswal/llama3b-econometrics", help="Name of Hugging Face repository")
    args = parser.parse_args()
    upload_model(args.repo_name)
