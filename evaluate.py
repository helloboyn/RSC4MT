import sacrebleu
import pandas as pd

DOMAINS = [
    ("Agriculture", 200),
    ("Tourism", 200),
    ("Governance", 200),
    ("Climate", 200),
    ("Healthcare", 200),
    ("Science_Technology", 200),
    ("Judiciary", 200),
    ("Education", 600)
]

REFERENCE_FILE = "data/test_bengali.txt"
PREDICTION_FILE = "predictions.txt"
OUTPUT_FILE = "domain_scores.csv"

def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def evaluate(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs])
    ter = sacrebleu.corpus_ter(preds, [refs])

    return {
        "BLEU": round(bleu.score, 2),
        "chrF": round(chrf.score, 2),
        "TER": round(ter.score, 2)
    }

def main():
    preds = read_lines(PREDICTION_FILE)
    refs = read_lines(REFERENCE_FILE)

    assert len(preds) == len(refs)

    results = []
    start = 0

    for name, size in DOMAINS:
        end = start + size

        p = preds[start:end]
        r = refs[start:end]

        m = evaluate(p, r)

        results.append({
            "Domain": name,
            "Samples": len(p),
            "BLEU": m["BLEU"],
            "chrF": m["chrF"],
            "TER": m["TER"]
        })

        start = end

    overall = evaluate(preds, refs)

    results.append({
        "Domain": "OVERALL",
        "Samples": len(preds),
        "BLEU": overall["BLEU"],
        "chrF": overall["chrF"],
        "TER": overall["TER"]
    })

    df = pd.DataFrame(results)

    print(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()