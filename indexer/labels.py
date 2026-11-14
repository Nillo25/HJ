"""Нормализация подписей комнат и ручная основа словаря."""
import re

# Ручная основа. Всё, что не нашлось здесь, собирается в unknown_labels.txt
# и разрешается одним пакетным вызовом модели (resolve_labels.py).
SEED = {
    # финские (cubicasa5k)
    "mh": "bedroom", "makuuhuone": "bedroom",
    "oh": "living", "olohuone": "living",
    "k": "kitchen", "keittio": "kitchen", "keittiö": "kitchen",
    "kk": "kitchen", "keittokomero": "kitchen",
    "kh": "bathroom", "kph": "bathroom", "ph": "bathroom",
    "pesuh": "bathroom", "pesuhuone": "bathroom", "sh": "bathroom",
    "wc": "wc",
    "s": "sauna", "sauna": "sauna",
    "parveke": "balcony", "par": "balcony",
    "vh": "storage",          # гардеробная — НЕ спальня
    "et": "corridor", "eteinen": "corridor",
    "khh": "technical",       # kodinhoitohuone
    # немецкие / швейцарские (swiss_dwellings)
    "zimmer": "bedroom", "schlafzimmer": "bedroom",
    "kuche": "kitchen", "küche": "kitchen",
    "bad": "bathroom", "dusche": "bathroom",
    "korridor": "corridor", "flur": "corridor", "gang": "corridor",
    "wohnen": "living", "wohnen/essen": "living", "wohnzimmer": "living",
    "balkon": "balcony", "loggia": "balcony", "terrasse": "balcony",
    "reduit": "storage", "abstellraum": "storage", "keller": "storage",
    "treppenhaus": "corridor",
    "technik": "technical", "technikraum": "technical",
    # французские
    "cuisine": "kitchen", "chambre": "bedroom", "couloir": "corridor",
    "salle de bains": "bathroom", "sejour": "living", "séjour": "living",
    # итальянские
    "cucina": "kitchen", "stanza": "bedroom", "bagno": "bathroom",
    "corridoio": "corridor", "soggiorno": "living",
    # эстонские (ifc_models)
    "tualettruum": "wc", "magamistuba": "bedroom", "koogitoa": "kitchen",
    "elutuba": "living", "koridor": "corridor",
    # английские
    "bedroom": "bedroom", "kitchen": "kitchen", "bathroom": "bathroom",
    "toilet": "wc", "restroom": "wc", "corridor": "corridor",
    "living room": "living", "balcony": "balcony", "storage": "storage",
    "office": "office", "lobby": "corridor",
}

# Подпись вида «12.34 m²» — это печатная площадь, а не название комнаты
AREA_RE = re.compile(r"^\d+[.,]\d+\s*m[²2]$", re.I)
# Подпись вида «2.10» или «105» — номер комнаты
NUMBER_RE = re.compile(r"^\d+([.\-]\d+)*$")


def classify_text(s: str) -> str:
    """area | number | name | other — что это за подпись на чертеже."""
    t = (s or "").strip()
    if not t:
        return "other"
    if AREA_RE.match(t):
        return "area"
    if NUMBER_RE.match(t):
        return "number"
    return "name"


def norm(s: str) -> str:
    """Нормализация подписи для поиска по словарю.

    'MH 2.03' -> 'mh'   'KPH.' -> 'kph'   'Küche' -> 'küche'
    """
    t = (s or "").strip().lower()
    t = re.sub(r"\s*\d+([.\-]\d+)*\s*$", "", t)   # хвостовой номер комнаты
    t = t.strip(" .,:;()[]")
    t = re.sub(r"\s+", " ", t)
    return t


def subtype_for(label: str, dict_rows: dict | None = None) -> str | None:
    """Смысловой тип по подписи. None = неизвестно, пойдёт в unknown_labels."""
    n = norm(label)
    if not n:
        return None
    if dict_rows and n in dict_rows:
        return dict_rows[n]
    return SEED.get(n)


def load_dict(con) -> dict:
    """Словарь из базы поверх SEED."""
    rows = dict(SEED)
    for ln, st in con.execute("SELECT label_norm, subtype FROM label_dict"):
        rows[ln] = st
    return rows


def seed_dict(con):
    con.executemany(
        "INSERT OR IGNORE INTO label_dict(label_norm, subtype, source) VALUES (?,?,'manual')",
        [(k, v) for k, v in SEED.items()],
    )
