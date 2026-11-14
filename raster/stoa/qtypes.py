"""Question text (EN/FI) -> canonical question type."""
import re

# order matters: first match wins
PATTERNS = [
    ("windows",   r"\bwindows\b|ikkuna"),
    ("doors",     r"\bdoors\b|\bovea\b"),
    ("toilets",   r"toilet bowls|wc-istuin"),
    ("sinks",     r"sinks|tiskiallas|pesu- ja tiski"),
    ("showers",   r"showers|suihku"),
    ("saunas",    r"saunas|saunaa"),
    ("kitchens",  r"kitchens|keittiö"),
    ("bedrooms",  r"bedrooms|makuuhuone"),
    ("bathrooms", r"bathrooms|kylpy- tai pesuhuone"),  # before wc_rooms: text mentions "separate WC"
    ("wc_rooms",  r"separate wc|erillistä wc"),
    ("balconies", r"balcon|parveke"),
    ("floors",    r"floors \(storeys\)|kerrosta"),
]
QTYPES = [p[0] for p in PATTERNS]


def qtype_of(question: str) -> str:
    q = question.lower()
    for name, pat in PATTERNS:
        if re.search(pat, q):
            return name
    raise ValueError(f"unknown question type: {question!r}")


def subtype_of(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if name.startswith("high_quality_architectural"):
        return "hq_arch"
    if name.startswith("high_quality"):
        return "hq"
    if name.startswith("colorful"):
        return "colorful"
    return "other"
