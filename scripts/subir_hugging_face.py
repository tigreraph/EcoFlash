from huggingface_hub import HfApi

api = HfApi()
repo_id = "tigreraph/ecoflash-dataset"

api.upload_folder(
    folder_path="dataset",
    repo_id=repo_id,
    repo_type="dataset"
)

print("✔ Dataset subido con éxito.")
