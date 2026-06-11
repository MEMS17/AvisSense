# AvisSense

Analyse de sentiment d'avis en français avec CamemBERT fine-tuné sur le dataset Allociné.

**Projet de fin de module M106 - Deep Learning**

## Demo

[🤗 Hugging Face Space — Avis Sense](https://huggingface.co/spaces/rvss311/avis_sense)

Entrez un avis en français et l'application prédit s'il est **POSITIF** ou **NÉGATIF** avec un score de confiance.

## Stack technique

- **Modèle :** CamemBERT (`camembert-base`) fine-tuné sur 5000 avis Allociné
- **Framework :** PyTorch + Hugging Face Transformers
- **Interface :** Gradio
- **Déploiement :** Hugging Face Spaces (CPU)
- **Dataset :** [Allociné](https://huggingface.co/datasets/allocine) via Hugging Face Datasets

## Résultats

- **Entraînement :** 5000 échantillons, 1 epoch, batch size 8
- **Validation :** ~85% accuracy (estimé)
- **Exemple :** "Super film, je le recommande" → POSITIF (98% de confiance)

## Arborescence

```text
avis-sense/
├── app.py                 # Interface Gradio
├── requirements.txt       # Dépendances
├── .gitignore
├── src/
│   ├── predict.py         # Inférence avec CamemBERT
│   ├── train.py           # Script d'entraînement
│   ├── evaluate.py        # Évaluation
│   └── utils.py           # Utilitaires
├── notebooks/
│   └── avis_sense_training.ipynb  # Notebook d'entraînement
├── model/
│   └── saved_model/       # Poids du modèle fine-tuné
└── docs/
    └── project_plan.md
```

## Installation

```bash
uv init
uv add gradio torch transformers datasets scikit-learn pandas numpy accelerate sentencepiece protobuf
```

## Lancement local

```bash
uv run python app.py
```

- Fine-tuning CamemBERT, déploiement HF Spaces, interface Gradio
