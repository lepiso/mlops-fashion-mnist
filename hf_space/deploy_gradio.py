import argparse
import os
import shutil
from huggingface_hub import HfApi

def deploy_gradio(username, space_name="fashion-mnist-gradio",
            deploy_dir="/tmp/hf_gradio"):

    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)

    shutil.copy2("hf_space/gradio_app.py", f"{deploy_dir}/app.py")
    print("   Copie : hf_space/gradio_app.py")

    requirements = """gradio
requests
Pillow
numpy
matplotlib
"""
    with open(f"{deploy_dir}/requirements.txt", "w") as f:
        f.write(requirements)

    readme = f"""---
title: Fashion MNIST MLOps
emoji: 👗
colorFrom: purple
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Fashion MNIST MLOps Pipeline
Classification de vêtements — 10 catégories
"""
    with open(f"{deploy_dir}/README.md", "w") as f:
        f.write(readme)

    print(f"Dossier prêt : {deploy_dir}")

    api     = HfApi()
    repo_id = f"{username}/{space_name}"
    print(f"Création du Space : {repo_id}")
    try:
        api.create_repo(
            repo_id=space_name,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True
        )
        print("Space créé !")
    except Exception as e:
        print(f"   {e}")

    print("Upload en cours...")
    api.upload_folder(
        folder_path=deploy_dir,
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy Fashion MNIST Gradio MLOps"
    )
    print(f"""
{'='*55}
DEPLOIEMENT REUSSI !
App : https://huggingface.co/spaces/{repo_id}
URL : https://{username}-{space_name}.hf.space
{'='*55}
""")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    print("="*55)
    print("Déploiement Gradio → HuggingFace Spaces")
    print("="*55)
    deploy_gradio(args.username)