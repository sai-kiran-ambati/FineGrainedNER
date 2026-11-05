from string import punctuation


# Telugu/Unicode punctuation that can appear as tokens
PUNCTUATION = set(
    punctuation
    + "।॥,.:;!?()[]{}\"'“”‘’"
)


def is_punctuation_token(token):
    """Return True if the token is only punctuation."""
    return all(char in PUNCTUATION for char in token)


def filter_candidates(candidates):
    """
    Remove candidates that are obviously not useful entity candidates.

    Important:
    This filtering is applied AFTER candidate generation.
    The original candidate generator remains unchanged.
    """

    filtered = []

    for candidate in candidates:

        tokens = candidate["text"].split()

        # Remove spans consisting entirely of punctuation
        if all(is_punctuation_token(token) for token in tokens):
            continue

        # Remove candidates whose first token is punctuation
        if is_punctuation_token(tokens[0]):
            continue

        # Remove candidates whose last token is punctuation
        if is_punctuation_token(tokens[-1]):
            continue

        filtered.append(candidate)

    return filtered