import os
import argparse
import numpy as np
import pandas as pd

FASHION_LABELS = {
    0: "T-shirt/top", 1: "Trouser",  2: "Pullover",
    3: "Dress",       4: "Coat",     5: "Sandal",
    6: "Shirt",       7: "Sneaker",  8: "Bag",
    9: "Ankle boot"
}

def prepare_dataset(csv_path="data/raw/fashion-mnist_train.csv",
                    save_path="data/raw/dataset.csv",
                    sample_size=None):
    print(f"📂 Lecture : {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"   Shape originale : {df.shape}")

    df = df.rename(columns={"label": "target"})
    old_cols = [f"pixel{i}" for i in range(1, 785)]
    new_cols = [f"pixel_{i}" for i in range(784)]
    df = df.rename(columns=dict(zip(old_cols, new_cols)))

    for col in new_cols:
        df[col] = df[col] / 255.0

    df["label_name"] = df["target"].map(FASHION_LABELS)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"   Sous-échantillon : {sample_size} images")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"\n📊 Distribution des classes :")
    for cls_id, count in df["target"].value_counts().sort_index().items():
        print(f"   {cls_id} - {FASHION_LABELS[cls_id]:<15} : {count}")
    print(f"\n✅ Sauvegardé : {save_path}  |  Shape : {df.shape}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/fashion-mnist_train.csv")
    parser.add_argument("--output", default="data/raw/dataset.csv")
    parser.add_argument("--sample", type=int, default=10000)
    args = parser.parse_args()
    prepare_dataset(csv_path=args.input, save_path=args.output, sample_size=args.sample)