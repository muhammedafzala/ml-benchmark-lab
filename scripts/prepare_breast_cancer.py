"""Export scikit-learn's bundled diagnostic breast cancer data as a local CSV."""

from pathlib import Path

from sklearn.datasets import load_breast_cancer


def main() -> None:
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    frame = load_breast_cancer(as_frame=True).frame
    output = raw / "breast_cancer.csv"
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame):,} rows to {output}")


if __name__ == "__main__":
    main()
