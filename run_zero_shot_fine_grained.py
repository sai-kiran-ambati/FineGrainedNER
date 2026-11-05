"""Fine-grained PromptNER without demonstrations (zero-shot)."""

import argparse

from run_20_sentences_fine_grained import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", type=int, default=20)
    args = parser.parse_args()
    main(num_sentences=args.sentences, ablation="zero_shot")
