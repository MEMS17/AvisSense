// script.js — Logique du front minimal : appelle POST /predict et affiche
// le résultat. C'est la consigne du prof en une page :
// « colle un avis -> positif/négatif + confiance ».

// ── Récupération des éléments HTML qu'on va manipuler ──────────────────────
const reviewInput = document.getElementById("review-input");
const analyzeBtn = document.getElementById("analyze-btn");
const resultBox = document.getElementById("result");
const resultLabel = document.getElementById("result-label");
const confidenceBar = document.getElementById("confidence-bar");
const resultConfidence = document.getElementById("result-confidence");
const errorBox = document.getElementById("error");

/** Affiche un message d'erreur et cache le résultat précédent. */
function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  resultBox.classList.add("hidden");
}

/** Affiche la prédiction : label coloré + barre de confiance animée. */
function showResult(label, confidence) {
  errorBox.classList.add("hidden");
  resultBox.classList.remove("hidden");

  const isPositive = label === "positif";
  resultLabel.textContent = isPositive ? "😊 Avis positif" : "😞 Avis négatif";
  // On pose la classe CSS "positif" ou "negatif" -> couleur verte ou rouge
  resultLabel.className = "result-label " + (isPositive ? "positif" : "negatif");

  // La confiance (0 à 1) devient une largeur en % pour la barre
  const percent = (confidence * 100).toFixed(1);
  confidenceBar.style.width = percent + "%";
  confidenceBar.style.background = isPositive ? "#16a34a" : "#dc2626";
  resultConfidence.textContent = `Confiance du modèle : ${percent} %`;
}

/** Envoie le texte à l'endpoint POST /predict de l'API FastAPI. */
async function analyzeReview() {
  const text = reviewInput.value.trim();

  // Validation côté client : inutile d'appeler l'API pour un texte vide
  // (l'API revérifie de toute façon côté serveur — défense en profondeur)
  if (!text) {
    showError("Veuillez saisir un avis avant d'analyser.");
    return;
  }

  // Pendant la requête : bouton désactivé (évite les doubles clics) et
  // libellé explicite pour l'utilisateur
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyse en cours...";

  try {
    // fetch en URL RELATIVE ("/predict" et pas "http://...") : le front est
    // servi par la même application que l'API, donc même domaine, que ce
    // soit en local (127.0.0.1:8000) ou sur le Space déployé.
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    });

    if (!response.ok) {
      // En cas d'erreur, FastAPI renvoie {"detail": "message explicite"}
      const error = await response.json();
      showError(error.detail || "Erreur lors de la prédiction.");
      return;
    }

    // Réponse de l'API : {"label": "positif", "confidence": 0.94, ...}
    const data = await response.json();
    showResult(data.label, data.confidence);
  } catch (err) {
    // fetch a échoué avant même d'avoir une réponse HTTP : serveur éteint,
    // réseau coupé, Space en cours de démarrage...
    showError("Impossible de contacter l'API. Le serveur est-il lancé ?");
  } finally {
    // Quoi qu'il arrive (succès OU erreur), on réactive le bouton
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyser";
  }
}

// Clic sur le bouton -> analyse
analyzeBtn.addEventListener("click", analyzeReview);

// Confort : Ctrl+Entrée dans la zone de texte lance aussi l'analyse
reviewInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && event.ctrlKey) analyzeReview();
});
