"""Classify an email as spam or ham.

Examples:
  python predict.py --subject "You won!!!" --body "Click http://prize.example to claim $1000"
  python predict.py --file message.eml
  python predict.py --file message.txt
"""
import argparse
import joblib

from text_utils import combine, read_eml, spam_score


def classify(model, subject: str, body: str):
    score = float(spam_score(model, [combine(subject, body)])[0])
    return ("SPAM" if score >= 0.5 else "HAM"), score


def main():
    p = argparse.ArgumentParser(description="Email spam classifier")
    p.add_argument("--subject", default="")
    p.add_argument("--body", default="")
    p.add_argument("--file", help=".eml or plain-text file (first line used as subject for .txt)")
    a = p.parse_args()

    subject, body = a.subject, a.body
    if a.file:
        if a.file.lower().endswith(".eml"):
            subject, body = read_eml(a.file)
        else:
            lines = open(a.file, encoding="utf-8", errors="ignore").read().splitlines()
            subject, body = (lines[0], "\n".join(lines[1:])) if lines else ("", "")
    if not (subject or body):
        p.error("provide --subject/--body or --file")

    model = joblib.load("models/spam_model.joblib")
    label, score = classify(model, subject, body)
    print(f"Result: {label}  (spam score {score:.1%})")


if __name__ == "__main__":
    main()
