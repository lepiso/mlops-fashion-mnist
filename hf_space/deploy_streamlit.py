import argparse
import os
import shutil
from huggingface_hub import HfApi


def deploy_streamlit(username, space_name="fashion-mnist-app",
                deploy_dir="/tmp/hf_streamlit"):
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)

    shutil.copy2("hf_space/streamlit_hf.py", f"{deploy_dir}/app.py")
    print("   Copie : hf_space/streamlit_hf.py")

    requirements = """streamlit
requests
Pillow
numpy
matplotlib
"""
    with open(f"{deploy_dir}/requirements.txt", "w") as f:
        f.write(requirements)
    print("   Création : requirements.txt")

    readme = f"""---
title: Fashion MNIST MLOps App
emoji: 👗
colorFrom: purple
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# Fashion MNIST MLOps App
"""
    with open(f"{deploy_dir}/README.md", "w") as f:
        f.write(readme)
    print("   Création : README.md")
    print(f"Dossier prêt : {deploy_dir}")

    api = HfApi()
    repo_id = f"{username}/{space_name}"
    print(f"Création du Space : {repo_id}")
    try:
        api.create_repo(
            repo_id=space_name,
            repo_type="space",
            space_sdk="streamlit",
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
        commit_message="Deploy Fashion MNIST Streamlit MLOps"
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
    parser.add_argument("--space-name", default="fashion-mnist-app")
    parser.add_argument("--deploy-dir", default="/tmp/hf_streamlit")
    args = parser.parse_args()

    print("="*55)
    print("Déploiement Streamlit → HuggingFace Spaces")
    print("="*55)

    if not os.path.exists("app/streamlit_app.py"):
        print("ERREUR : app/streamlit_app.py introuvable !")
        exit(1)

    deploy_streamlit(args.username, args.space_name, args.deploy_dir)