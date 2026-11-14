"""Room label normalisation and the hand-written dictionary."""
import re

# Hand-written seed. Anything missing goes to unknown_labels.txt and is
# resolved in one batch LLM call (resolve_labels.py).
SEED = {
    # Finnish (cubicasa5k)
    "mh": "bedroom", "makuuhuone": "bedroom",
    "oh": "living", "olohuone": "living",
    "k": "kitchen", "keittio": "kitchen", "keittiö": "kitchen",
    "kk": "kitchen", "keittokomero": "kitchen",
    "kh": "bathroom", "kph": "bathroom", "ph": "bathroom",
    "pesuh": "bathroom", "pesuhuone": "bathroom", "sh": "bathroom",
    "wc": "wc",
    "s": "sauna", "sauna": "sauna",
    "parveke": "balcony", "par": "balcony",
    "vh": "storage",          # walk-in closet, NOT a bedroom
    "et": "corridor", "eteinen": "corridor",
    "khh": "technical",       # kodinhoitohuone
    # German / Swiss (swiss_dwellings)
    "zimmer": "bedroom", "schlafzimmer": "bedroom",
    "kuche": "kitchen", "küche": "kitchen",
    "bad": "bathroom", "dusche": "bathroom",
    "korridor": "corridor", "flur": "corridor", "gang": "corridor",
    "wohnen": "living", "wohnen/essen": "living", "wohnzimmer": "living",
    "balkon": "balcony", "loggia": "balcony", "terrasse": "balcony",
    "balcone": "balcony", "terrazza": "balcony", "veranda": "balcony",
    "balcon": "balcony", "terrasson": "balcony", "sitzplatz": "balcony",
    "gartensitzplatz": "balcony", "patio": "balcony", "dachterrasse": "balcony",
    "reduit": "storage", "abstellraum": "storage", "keller": "storage",
    "treppenhaus": "corridor",
    "technik": "technical", "technikraum": "technical",
    # French
    "cuisine": "kitchen", "chambre": "bedroom", "couloir": "corridor",
    "salle de bains": "bathroom", "sejour": "living", "séjour": "living",
    # Italian
    "cucina": "kitchen", "stanza": "bedroom", "bagno": "bathroom",
    "corridoio": "corridor", "soggiorno": "living",
    # Estonian (ifc_models)
    "tualettruum": "wc", "magamistuba": "bedroom", "koogitoa": "kitchen",
    "elutuba": "living", "koridor": "corridor",
    # English
    "bedroom": "bedroom", "kitchen": "kitchen", "bathroom": "bathroom",
    "toilet": "wc", "restroom": "wc", "corridor": "corridor",
    "living room": "living", "balcony": "balcony", "storage": "storage",
    "office": "office", "lobby": "corridor",
}

# A label like "12.34 m2" is a printed area, not a room name
AREA_RE = re.compile(r"^\d+[.,]\d+\s*m[²2]$", re.I)
# A label like "2.10" or "105" is a room number
NUMBER_RE = re.compile(r"^\d+([.\-]\d+)*$")


def classify_text(s: str) -> str:
    """area | number | name | other - what kind of label this is."""
    t = (s or "").strip()
    if not t:
        return "other"
    if AREA_RE.match(t):
        return "area"
    if NUMBER_RE.match(t):
        return "number"
    return "name"


def norm(s: str) -> str:
    """Normalise a label for dictionary lookup.

    'MH 2.03' -> 'mh'   'KPH.' -> 'kph'   'Küche' -> 'küche'
    """
    t = (s or "").strip().lower()
    t = re.sub(r"\s*\d+([.\-]\d+)*\s*$", "", t)   # trailing room number
    t = t.strip(" .,:;()[]")
    t = re.sub(r"\s+", " ", t)
    return t


def subtype_for(label: str, dict_rows: dict | None = None) -> str | None:
    """Semantic type for a label. None means unknown -> unknown_labels."""
    n = norm(label)
    if not n:
        return None
    if dict_rows and n in dict_rows:
        return dict_rows[n]
    return SEED.get(n)


def load_dict(con) -> dict:
    """Dictionary rows from the database layered over SEED."""
    rows = dict(SEED)
    for ln, st in con.execute("SELECT label_norm, subtype FROM label_dict"):
        rows[ln] = st
    return rows


def seed_dict(con):
    con.executemany(
        "INSERT OR IGNORE INTO label_dict(label_norm, subtype, source) VALUES (?,?,'manual')",
        [(k, v) for k, v in SEED.items()],
    )


# Outdoor spaces: excluded from apartment area and from "numbered indoor
# rooms". Verified on swiss_dwellings: the area mismatch was exactly
# balconies plus unlabelled slivers.
OUTDOOR = {"balcony"}


def is_outdoor(subtype: str | None) -> bool:
    return subtype in OUTDOOR
