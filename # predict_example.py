# predict_example.py
import joblib
import argparse
import pandas as pd
from emotion_detector import preprocess_text

def predict_texts(model_path, vectorizer_path, texts):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    cleaned = [preprocess_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return preds, probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--vectorizer', required=True)
    parser.add_argument('--examples', nargs='+', required=False,
                        help='Example texts. If omitted, uses a small default list.')
    args = parser.parse_args()

    examples = args.examples or [
        "I can't believe I won!",
        "Everything is falling apart around me.",
        "This is disgusting, I can't stand it.",
        "I'm shaking, this is terrifying.",
        "I'm so mad at what happened today!",
        "That's unbelievable, I didn't expect it!"
    ]

    preds, probs = predict_texts(args.model, args.vectorizer, examples)
    for i, t in enumerate(examples):
        print(f"TEXT: {t}")
        print(f"PRED: {preds[i]}")
        if probs is not None:
            # show top-2 label probabilities for readability
            top_idx = probs[i].argsort()[::-1][:2]
            labels = model.classes_
            top = [(labels[idx], float(probs[i][idx])) for idx in top_idx]
            print("TOP PROBS:", top)
        print("-" * 60)
