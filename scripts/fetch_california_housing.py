"""Download California Housing and export the CSV used by the benchmark."""

from pathlib import Path

from sklearn.datasets import fetch_california_housing


def main() -> None:
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    housing = fetch_california_housing(data_home=raw / "sklearn-cache", as_frame=True)
    output = raw / "california_housing.csv"
    housing.frame.to_csv(output, index=False)
    print(f"Saved {len(housing.frame):,} rows to {output}")


if __name__ == "__main__":
    main()
