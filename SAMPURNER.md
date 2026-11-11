# PromptNER on SampurNER Telugu

SampurNER already provides fine-grained BIO annotations and official train,
validation, and test splits. These scripts keep the SampurNER labels and data
separate from the anonymous TeluguNER files.

## Prepare the Telugu split

Install dependencies, then download and convert the official Telugu (`te`)
CoNLL files:

```bash
pip install -r requirements.txt
python prepare_sampurner.py
```

The script writes `data/sampurner/{train,validation,test}.json`, using the
repository's `SampurNER_te_{train,dev,test}.txt` files. Use `--language` to
prepare another language code.

## Run PromptNER

Fine-grained labels (the default):

```bash
python run_sampurner_promptner.py --level fine --sentences 20
```

Coarse labels, using the parent category before the first hyphen (for example,
`person-actor` → `PERSON`):

```bash
python run_sampurner_promptner.py --level coarse --sentences 20
```

OpenAI is the default provider. Set `NER_PROVIDER=sarvam` to use Sarvam. The
same `OPENAI_API_KEY`, `SARVAM_API_KEY`, `GPT5_MODEL`, and `SARVAM_MODEL`
environment variables used by the existing runners apply here. Sarvam uses
three concurrent requests by default; set `NER_MAX_CONCURRENCY=1` to run
sequentially or use a lower value if you hit rate limits. `--split`
accepts `train`, `validation`, or `test`; `--few-shot 0` disables training
examples, and `--max-span-tokens` controls candidate span length.

Results are written under `results/` with SampurNER, provider, level, split,
and sentence count in the file name.

Each run now reports and saves three matchers: token-wise BIO tagging (outside
tokens excluded from entity precision/recall/F1), exact entity-wise span plus
label, and partial matching (any span overlap plus the same label, matched
one-to-one). The main run JSON contains the per-sentence scores and aggregate
micro scores; a second `sampurner_matching_comparison_...json` file contains
the comparison summary and per-sentence metrics.

To generate a comparison file from a previously saved run without calling the
model again:

```bash
python compare_sampurner_results.py --input results/promptner_sampurner_sarvam_fine_test20.json
```

The coarse grouping is SampurNER's own hierarchy: the prefix of each native
fine label. It is intentionally not projected onto the anonymous dataset's
PER/LOC/ORG label inventory.

Dataset card and citation: [prachuryyaIITG/SampurNER](https://huggingface.co/datasets/prachuryyaIITG/SampurNER).
