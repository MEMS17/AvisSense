import gradio as gr
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CamembertTokenizer.from_pretrained("./model/saved_model")
model = CamembertForSequenceClassification.from_pretrained("./model/saved_model").to(device)
model.eval()

def predict_sentiment(text):
    if not text or not text.strip():
        return "Veuillez saisir un avis à analyser."
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()
    label = "POSITIF" if pred == 1 else "NEGATIF"
    return f"{label} (confiance: {conf:.1%})"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, label="Avis en français"),
    outputs=gr.Textbox(label="Résultat"),
    title="AvisSense",
    description="Analyse de sentiment d'avis en français (CamemBERT)",
)

if __name__ == "__main__":
    demo.launch()
