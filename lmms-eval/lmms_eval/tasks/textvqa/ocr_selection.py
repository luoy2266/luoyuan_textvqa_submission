import re


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}

NUMERIC_QUESTION_WORDS = {
    "amount",
    "cost",
    "date",
    "digit",
    "digits",
    "hour",
    "many",
    "month",
    "number",
    "phone",
    "price",
    "score",
    "time",
    "year",
}

URL_QUESTION_WORDS = {"site", "url", "web", "website"}
ADDRESS_QUESTION_WORDS = {"address", "avenue", "road", "street"}


def normalize_text(text):
    text = str(text).casefold()
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_tokens(ocr_tokens):
    return [str(token).strip() for token in (ocr_tokens or []) if str(token).strip()]


def _dedupe_tokens(tokens):
    seen = set()
    result = []
    for token in tokens:
        key = normalize_text(token) or token.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(token)
    return result


def _question_terms(question):
    return {
        word
        for word in normalize_text(question).split()
        if len(word) > 1 and word not in STOP_WORDS
    }


def _char_ngrams(text, n=3):
    compact = text.replace(" ", "")
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def _shape_bonus(token, question_terms):
    token_lower = token.casefold()
    token_norm = normalize_text(token)
    score = 0
    if question_terms & NUMERIC_QUESTION_WORDS and re.search(r"\d", token_norm):
        score += 18
    if question_terms & URL_QUESTION_WORDS and (
        "." in token_lower or "www" in token_lower or "http" in token_lower
    ):
        score += 18
    if question_terms & ADDRESS_QUESTION_WORDS and re.search(r"\d", token_norm):
        score += 10
    if {"price", "cost", "amount"} & question_terms and (
        "$" in token_lower or re.search(r"\d+[.,]\d{2}", token_norm)
    ):
        score += 14
    return score


def _question_aware_tokens(question, tokens):
    unique_tokens = _dedupe_tokens(tokens)
    question_norm = normalize_text(question)
    terms = _question_terms(question)
    question_grams = _char_ngrams(question_norm)

    scored = []
    for idx, token in enumerate(unique_tokens):
        token_norm = normalize_text(token)
        token_terms = token_norm.split()
        score = 0

        if token_norm and token_norm in question_norm:
            score += 50
        if token_norm in terms:
            score += 80

        for term in terms:
            if len(term) <= 2 or not token_norm:
                continue
            if term in token_norm or token_norm in term:
                score += 25

        for token_term in token_terms:
            if token_term in terms:
                score += 40

        overlap = len(_char_ngrams(token_norm) & question_grams)
        score += min(overlap * 2, 20)
        score += _shape_bonus(token, terms)

        scored.append((score, idx, token))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [token for _, _, token in scored]


def select_ocr_tokens(question, ocr_tokens, strategy="first", max_tokens=16):
    tokens = _clean_tokens(ocr_tokens)
    strategy = (strategy or "first").lower()

    if strategy in {"first", "ordered", "first_n"}:
        selected = tokens
    elif strategy in {"unique", "dedupe", "dedup"}:
        selected = _dedupe_tokens(tokens)
    elif strategy in {"question_aware", "qaware", "question-aware"}:
        selected = _question_aware_tokens(question, tokens)
    else:
        raise ValueError(f"Unknown OCR selection strategy: {strategy}")

    max_tokens = int(max_tokens)
    if max_tokens > 0:
        return selected[:max_tokens]
    return selected
