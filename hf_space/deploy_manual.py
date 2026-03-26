import argparse
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi

def check_files():
    required = [
        "models/model.pkl",
        "models/feature_names.pkl",
        "src/api/main.py",
        "src/api/schemas.py",
        "configs/config.yaml",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("Fichiers manquants :")
        for f in missing:
            print(f"   - {f}")
        return False
    print("Tous les fichiers sont presents")
    return True

def prepare_deploy_folder(deploy_dir="/tmp/hf_fashion_mnist"):
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(f"{deploy_dir}/src/api")
    os.makedirs(f"{deploy_dir}/models")
    os.makedirs(f"{deploy_dir}/configs")

    copies = [
        ("src/api/main.py",          f"{deploy_dir}/src/api/main.py"),
        ("src/api/schemas.py",       f"{deploy_dir}/src/api/schemas.py"),
        ("configs/config.yaml",      f"{deploy_dir}/configs/config.yaml"),
        ("models/model.pkl",         f"{deploy_dir}/models/model.pkl"),
        ("models/feature_names.pkl", f"{deploy_dir}/models/feature_names.pkl"),
        ("models/label_names.pkl",   f"{deploy_dir}/models/label_names.pkl"),
    ]
    for src, dst in copies:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"   Copie : {src}")

    Path(f"{deploy_dir}/src/__init__.py").touch()
    Path(f"{deploy_dir}/src/api/__init__.py").touch()

    # Dockerfile
    dockerfile = """FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --no-cache-dir \\
    scikit-learn==1.4.0 pandas==2.2.0 numpy==1.26.4 joblib==1.3.2 \\
    fastapi==0.109.2 uvicorn[standard]==0.27.1 pydantic==2.6.1 \\
    PyYAML==6.0.1
ENV PYTHONPATH=/app
RUN useradd -m -u 1000 hfuser && chown -R hfuser:hfuser /app
USER hfuser
EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
"""
    with open(f"{deploy_dir}/Dockerfile", "w") as f:
        f.write(dockerfile)

    # README
    readme = """---
title: Fashion MNIST API
emoji: 👗
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Fashion MNIST MLOps API
Classification de vêtements — 10 catégories
"""
    with open(f"{deploy_dir}/README.md", "w") as f:
        f.write(readme)

    print(f"Dossier prêt : {deploy_dir}")
    return deploy_dir

def deploy(username, space_name, deploy_dir):
    api     = HfApi()
    repo_id = f"{username}/{space_name}"
    print(f"Création du Space : {repo_id}")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            space_hardware="cpu-basic",   
            private=False,
            exist_ok=True
        )
        print(f"Space créé : https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"   {e}")

    print(f"Upload en cours...")
    api.upload_folder(
        folder_path=deploy_dir,
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy Fashion MNIST FastAPI MLOps",
    )
    print(f"""
{'='*50}
DEPLOIEMENT REUSSI !
Space    : https://huggingface.co/spaces/{repo_id}
API Docs : https://{username}-{space_name}.hf.space/docs
Health   : https://{username}-{space_name}.hf.space/health
Predict  : https://{username}-{space_name}.hf.space/predict
Build en cours (~3-5 min)
{'='*50}
""")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username",   required=True)
    parser.add_argument("--space-name", default="fashion-mnist-api")
    parser.add_argument("--deploy-dir", default="/tmp/hf_fashion_mnist")
    args = parser.parse_args()

    print("="*50)
    print("Déploiement Fashion MNIST → HuggingFace Spaces")
    print("="*50)

    if not check_files():
        exit(1)

    deploy_dir = prepare_deploy_folder(args.deploy_dir)
    deploy(args.username, args.space_name, deploy_dir)