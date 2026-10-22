"""Small neural-network intent classifier: maps a sentence to a booking service.

It is a scikit-learn MLP on bag-of-words features. It is deliberately small so
the training step runs in a second inside the app.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

from booking_logic import SERVICES

# Starter dataset: several examples per service so the model works out of the box.
DEFAULT_EXAMPLES = [
    ("I need a haircut", "Haircut"), ("trim my hair please", "Haircut"),
    ("I want a new hairstyle", "Haircut"), ("my hair is too long", "Haircut"),
    ("My back hurts", "Massage"), ("I feel stressed and my shoulders are tight", "Massage"),
    ("I want to relax with a massage", "Massage"), ("my neck is sore", "Massage"),
    ("my tooth hurts", "Dental Checkup"), ("I need my teeth cleaned", "Dental Checkup"),
    ("I want a dentist checkup", "Dental Checkup"), ("I have a cavity", "Dental Checkup"),
    ("I need to see a doctor", "Doctor Consultation"), ("I feel sick and have a fever", "Doctor Consultation"),
    ("I have a headache and cough", "Doctor Consultation"), ("I need a medical consultation", "Doctor Consultation"),
    ("I want to lift weights", "Fitness Training"), ("I need a personal trainer", "Fitness Training"),
    ("help me get in shape", "Fitness Training"), ("I want to start working out", "Fitness Training"),
    ("I want a facial", "Beauty Treatment"), ("I need a spa day", "Beauty Treatment"),
    ("book me a manicure", "Beauty Treatment"), ("I want a skin treatment", "Beauty Treatment"),
]

CONFIDENCE_THRESHOLD = 0.40


def train(examples: list[tuple[str, str]]):
    """Train on (text, service) pairs. Returns (model, vectorizer)."""
    texts = [t for t, _ in examples]
    labels = [l for _, l in examples]
    if len(set(labels)) < 2:
        raise ValueError("Add examples for at least two different services before training.")
    vectorizer = CountVectorizer(lowercase=True)
    X = vectorizer.fit_transform(texts)
    model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42)
    model.fit(X, labels)
    return model, vectorizer


def predict(model, vectorizer, text: str):
    """Return (service, confidence). service is None when the model is not confident."""
    X = vectorizer.transform([text])
    if X.nnz == 0:  # none of the words were seen during training
        return None, 0.0
    probs = model.predict_proba(X)[0]
    best = probs.argmax()
    confidence = float(probs[best])
    service = str(model.classes_[best])
    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence
    return service, confidence
