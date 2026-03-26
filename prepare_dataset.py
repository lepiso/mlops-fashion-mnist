import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 1. Charger le dataset (déjà préparé)
# ─────────────────────────────────────────
df = pd.read_csv("data/raw/dataset.csv")

# ─────────────────────────────────────────
# 2. Infos générales
# ─────────────────────────────────────────
print(df.head())
print(df.info())
print(df.describe())
