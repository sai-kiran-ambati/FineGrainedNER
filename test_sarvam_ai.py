from pathlib import Path
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.build_prompt import build_prompt
from src.generate_token_candidates import (
    load_data,
    get_sentence,
    generate_token_candidates,
)


from src.filter_candidates import filter_candidates



# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_API_SUBSCRIPTION_KEY")

if not api_key:
    raise RuntimeError(
        "SARVAM_API_KEY (or SARVAM_API_SUBSCRIPTION_KEY) was not found in .env"
    )


# --------------------------------------------------
# Sarvam client (OpenAI-compatible Responses API)
# --------------------------------------------------

client = OpenAI(
    api_key=api_key,
    base_url="https://api.sarvam.ai/v2",
)


# --------------------------------------------------
# Load TeluguNER data
# --------------------------------------------------

df = load_data()


# --------------------------------------------------
# Select ONE test sentence
# --------------------------------------------------

sentence_id = 12735

sentence_df = get_sentence(
    df,
    sentence_id
)


# --------------------------------------------------
# Reconstruct sentence
# --------------------------------------------------

sentence = " ".join(
    sentence_df["word"].tolist()
)


# --------------------------------------------------
# Generate candidates
# --------------------------------------------------

all_candidates = generate_token_candidates(
    sentence_df,
    max_n=5
)

candidates = filter_candidates(
    all_candidates
)

print(f"Candidates before filtering: {len(all_candidates)}")
print(f"Candidates after filtering:  {len(candidates)}")


# --------------------------------------------------
# Build PromptNER prompt
# --------------------------------------------------

prompt = build_prompt(
    sentence,
    candidates
)


# --------------------------------------------------
# Display basic information
# --------------------------------------------------

print("=" * 80)
print("PROMPTNER SARVAM TEST")
print("=" * 80)

print(f"Sentence ID: {sentence_id}")
print(f"Number of candidates: {len(candidates)}")

print("\n" + "=" * 80)
print("INPUT SENTENCE")
print("=" * 80)

print(sentence)


# --------------------------------------------------
# Call Sarvam Responses API
# --------------------------------------------------

print("\n" + "=" * 80)
print("CALLING SARVAM")
print("=" * 80)

response = client.responses.create(
    model=os.getenv("SARVAM_MODEL", "sarvam-105b"),
    input=prompt,
    store=False,
)


# --------------------------------------------------
# Raw model output
# --------------------------------------------------

print("\n" + "=" * 80)
print("RAW SARVAM OUTPUT")
print("=" * 80)

print(response.output_text)
