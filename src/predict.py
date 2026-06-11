import argparse
from transformers import pipeline


MODEL_NAME = "cmarkea/distilcamembert-base-sentiment"

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME
)


def convert_label(label: str) -> str:
    """
    Convertit le label brut du modèle en label lisible pour l'utilisateur.
    Exemple : '5 stars' devient 'Positif'.
    """
    label = label.lower()

    if "1" in label or "2" in label:
        return "Négatif"
    if "3" in label:
        return "Neutre / mitigé"
    if "4" in label or "5" in label:
        return "Positif"

    return label


def predict(text: str):
    """
    Analyse le sentiment d'un texte français.
    """
    result = classifier(text)[0]

    raw_label = result["label"]
    sentiment = convert_label(raw_label)
    confidence = round(result["score"] * 100, 2)

    return sentiment, confidence, raw_label


def main():
    parser = argparse.ArgumentParser(
        description="AvisSense — Analyse de sentiment d'avis en français"
    )

    parser.add_argument(
        "text",
        nargs="?",
        default="Ce film est incroyable, les acteurs sont excellents.",
        help="Texte en français à analyser"
    )

    args = parser.parse_args()

    sentiment, confidence, raw_label = predict(args.text)

    print("=== AvisSense ===")
    print(f"Texte : {args.text}")
    print(f"Sentiment : {sentiment}")
    print(f"Confiance : {confidence} %")
    print(f"Label brut du modèle : {raw_label}")


if __name__ == "__main__":
    main()