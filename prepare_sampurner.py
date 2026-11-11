"""Download and convert SampurNER's official Telugu CoNLL splits."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "data" / "sampurner"
DATASET_ID = "prachuryyaIITG/SampurNER"


def convert_conll(path, split_name):
    """Convert blank-line-separated token<TAB>BIO rows to sentence records."""
    sentences = []
    tokens, tags = [], []

    def save_sentence():
        if not tokens:
            return
        entities = []
        start, label = None, None

        def flush(end):
            if start is not None:
                entities.append({
                    "text": " ".join(tokens[start:end]),
                    "label": label,
                    "start_token": start + 1,
                    "end_token": end,
                })

        for i, tag in enumerate(tags):
            if tag == "O" or "-" not in tag:
                flush(i)
                start, label = None, None
                continue
            prefix, current = tag.split("-", 1)
            if prefix == "B" or start is None or current != label:
                flush(i)
                start, label = i, current
        flush(len(tokens))
        sid = len(sentences)
        sentences.append({
            "sentence_id": f"{split_name}-{sid}",
            "text": " ".join(tokens),
            "tokens": list(tokens),
            "entities": entities,
        })
        tokens.clear()
        tags.clear()

    with path.open("r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, 1):
            if not raw.strip():
                save_sentence()
                continue
            line = raw.rstrip("\r\n")
            fields = line.split("\t")
            if len(fields) < 2:
                fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Malformed CoNLL line {line_number} in {path}: {raw!r}")
            tokens.append(fields[0])
            tags.append(fields[-1])
    save_sentence()
    return sentences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="te", help="SampurNER language code (default: te)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--revision", default="main", help="Hugging Face dataset revision")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install the `huggingface_hub` package first: pip install huggingface_hub") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source_split, target_split in (("train", "train"), ("dev", "validation"), ("test", "test")):
        filename = f"SampurNER_{args.language}_{source_split}.txt"
        local_path = hf_hub_download(
            repo_id=DATASET_ID,
            filename=filename,
            repo_type="dataset",
            revision=args.revision,
        )
        rows = convert_conll(Path(local_path), target_split)
        output = args.output_dir / f"{target_split}.json"
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{source_split}: {len(rows):,} sentences -> {output}")


if __name__ == "__main__":
    main()
