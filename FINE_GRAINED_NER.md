# Fine-grained Telugu PromptNER

The fine-grained runner keeps the existing TeluguNER data and BIO-derived gold
entities. It prompts the model with the 33 MultiCoNER v2 fine labels, their
MultiCoNER coarse parents, and the TeluguNER coarse definitions. Predictions
store the chosen fine label, its MultiCoNER parent, and the TeluguNER class used
for evaluation.

## Fine-label to TeluguNER mapping

| MultiCoNER v2 label group | MultiCoNER parent | TeluguNER class |
|---|---|---|
| Artist, Athlete, Cleric, Politician, Scientist, SportsManager, OtherPER | PER | PER |
| Facility, HumanSettlement, Station, OtherLOC | LOC | LOC |
| AerospaceManufacturer, CarManufacturer, MusicalGRP, ORG, PrivateCorp, PublicCorp, SportsGRP | GRP | ORG |
| Artwork, MusicalWork, Software, VisualWork, WrittenWork | CW | OTH |
| Clothing, Drink, Food, Vehicle, OtherProd | PROD | OTH |
| Disease, Symptom | MED | DIS |
| AnatomicalStructure, MedicalProcedure, Medication/Vaccine | MED | MED |

MultiCoNER v2 has no fine subtype for TeluguNER `CARDINAL`, `TIME`, or
`ORGANISM`; these are retained as direct TeluguNER outputs. `OTH` is also
available as a direct fallback for entities outside the MultiCoNER inventory.

The crosswalk is an explicit project mapping, not a claim that the taxonomies
are identical. In particular, `CW` and `PROD` map to the broad residual Telugu
class `OTH`.

## Run

OpenAI is the default provider:

```bash
python run_20_sentences_fine_grained.py
```

Use Sarvam:

```bash
NER_PROVIDER=sarvam python run_20_sentences_fine_grained.py
```

Sarvam runs use three concurrent requests by default to reduce time spent
waiting on network responses. If your Sarvam account rate-limits concurrent
calls, lower the setting; increase it gradually if requests remain stable:

```bash
NER_PROVIDER=sarvam NER_MAX_CONCURRENCY=2 python run_200_sentences_fine_grained.py
```

`NER_MAX_CONCURRENCY=1` restores sequential processing. Results remain in test
sentence order. Parallel requests can trigger provider throttling, so keep the
setting within your account's rate limits.

To process the first 50 test sentences and save a separate result file:

```bash
NER_PROVIDER=sarvam python run_50_sentences_fine_grained.py
```

This writes `results/promptner_sarvam_fine_grained50_results.json`.

The same runner supports 100, 200, 250, and 300 sentences, each with its own
script and result file:

```bash
NER_PROVIDER=sarvam python run_100_sentences_fine_grained.py
NER_PROVIDER=sarvam python run_200_sentences_fine_grained.py
NER_PROVIDER=sarvam python run_250_sentences_fine_grained.py
NER_PROVIDER=sarvam python run_300_sentences_fine_grained.py
```

For example, the 100-sentence run writes
`results/promptner_sarvam_fine_grained100_results.json`; the other output names
use the corresponding sentence count.

## Prompt ablation runs

Each ablation script defaults to the first 20 test sentences and accepts
`--sentences` to use another supported dataset size. Select the provider with
`NER_PROVIDER=sarvam` or `NER_PROVIDER=openai`.

```bash
NER_PROVIDER=sarvam python run_telugu_definitions_fine_grained.py --sentences 20
NER_PROVIDER=sarvam python run_zero_shot_fine_grained.py --sentences 20
NER_PROVIDER=sarvam python run_few_shot_fine_grained.py --sentences 20
```

The Telugu-definitions condition uses Telugu translations of all 33 fine-label
definitions and retains the existing coarse-label examples. Zero-shot uses
English fine-label definitions and omits demonstrations. Few-shot uses the
English fine-label definitions and includes the existing Telugu demonstrations.
Each condition saves to its own file, such as
`results/promptner_sarvam_zero_shot_fine_grained20_results.json`.

Results are written to a provider-specific file in `results/`. The runner
reports three matching approaches, all checking the predicted fine label's
TeluguNER coarse mapping against the original gold class:

1. **Token-wise:** BIO-tag micro F1 over entity tokens (outside `O` tokens are
   excluded from precision and recall); also reports token accuracy. Overlaps
   are resolved by preferring the longest predicted span, then the leftmost.
2. **Entity-wise:** exact start, end, and mapped TeluguNER class.
3. **Partial:** at least one token of span overlap and the same mapped class,
   with one-to-one prediction/gold matching.

The existing data has no gold fine-grained annotations, so these scores measure
coarse-class compatibility and span matching; they cannot measure whether the
fine subtype itself is correct.

To print the combined saved scores and each sentence's gold entities, true
positives, false positives, and false negatives without rerunning the model,
run `python src/fine_grained_evaluation.py`. It selects the most recently
updated fine-grained result file. Use `--provider sarvam` or `--provider openai`
to select a provider, and `--sentences N` to select a run size (20, 50, 100,
200, 250, or 300).

Each comparison also has a standalone report script:

```bash
python src/token_wise_matching.py --provider sarvam --sentences 50
python src/entity_wise_matching.py --provider sarvam --sentences 50
python src/partial_matching.py --provider sarvam --sentences 50
```

These scripts show the ground entities and true/false positives and negatives
for their matching method. `python src/analyze_errors_fine_grained.py` is an
alias for the combined evaluation report.
