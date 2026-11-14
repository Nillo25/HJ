"""Room label normalisation and the hand-written dictionary."""
import re
import unicodedata

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


def fold(s: str) -> str:
    """Lower case without accents: 'Réduit' -> 'reduit', 'Køkken' -> 'kokken', 'Küche' -> 'kuche'."""
    t = (s or "").lower()
    t = t.replace("ß", "ss").replace("æ", "ae").replace("ø", "o").replace("å", "a").replace("ł", "l")
    return "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))


# Word roots by meaning, over many languages (EN DE FR IT ES PT NL DA NO SV FI ET PL, plus the abbreviations the
# plans use). Matched as substrings of the folded label so that German-style compounds work ("Schlafzimmer",
# "Besprechungsraum", "Technikraum"). The order decides: a laundry is technical before "kuche" can make it a kitchen,
# a toilet is a WC before "bad" can make it a bathroom.
RULES = [
    ("technical", r"wasch(kuche|raum)|waschkuche|buanderie|lavanderia|tvattstuga|tvatt|vaskerom|laundry|"
                  r"technik|teknik|tekn|heiz|haustechnik|installation|schacht|skakt|shaft|aufzug|ascenseur|ascensore|"
                  r"\blift\b|elevator|elevaator|\bhiss\b|\bheis\b|server|\bit-?|ventil|elektr|\bel\b|\bel\.|mechanical|"
                  r"hvac|kjole|kole og|varmecentral|heizzentrale|\brlt\b|\bnshv\b|\bbmz\b|trafo|sprinkler|boiler|pump|"
                  r"affald|waste|muell|abfall|kodinhoito|\bkhh\b|hausanschluss|putzmittel|\bmech\b|"
                  r"verteil|schaltschrank|hovedtavle|\btavle\b|telerom|\bvent\b|vent\.|sluse|varmesentral|printer|\bkopi|pesukoda|koristus|soojus|veem|hooldus"),
    ("wc", r"\brwc\b|toilet|toalett|tualett|\bwc\b|\bwc[-\d]|[-\s]wc\b|\babort|restroom|lavatory|pissoir|urinal|inva.?wc|"
           r"aseo|servizi igienici|gabinete"),
    ("bathroom", r"bagno|\bbains?\b|salle d.?(eau|bain)|baderom|badezimmer|badevaerelse|badrum|badkamer|\bbad\b|"
                 r"duschraum|\bdusche|doccia|douche|vannituba|\bvann\b|kylpy|pesuh|\bkph\b|\bph\b|\bsh\b|"
                 r"bathroom|\bbath\b|shower|lazienka|\bbano\b|banheiro|sanitar|omkladning|umkleide|garderobe? (herre|dame)"),
    ("bedroom", r"schlaf|kinderzimmer|gastezimmer|\bzimmer\b|chambre|camera da letto|\bcamera\b|\bstanza\b|bedroom|"
                r"soverom|sovevaerelse|sovrum|magamis|makuu|dormitorio|habitacion|quarto|sypialnia|\bmh\b|\bpiece\b|"
                r"\blocale\b|\broom\b"),
    ("living", r"wohn|living|sejour|soggiorno|\bstue\b|olohuone|\boh\b|elutuba|vardagsrum|\bsalon\b|salle a manger|"
              r"esszimmer|\bessen\b|pranzo|\brepas\b|dining|stua\b|oleskelu|sala de estar|comedor|woonkamer"),
    ("kitchen", r"kuche|kitchen|cuisine|cucina|kjokken|kokken|\bkok\b|keitti|\bkook\b|koogi|cocina|cozinha|keuken|"
                r"kuchnia|kitchenette|pantry|teekoogi|\bkk\b"),
    ("storage", r"abstell|reduit|rumpelkammer|\blager|depot|ripostiglio|cantina|\bkeller\b|\bcave\b|storage|varasto|"
                r"\bladu\b|panipaik|almacen|trastero|bodega|\bkammer\b|archiv|boksamling|ropero|garderob|vaatehuone|"
                r"\bvh\b|\bgard\b|frys|\bkjol|torrvare|meieri|rotfrukt|soppel|varemottak|cabina armadio|dispensa|komero|\bclos\b|closet|stauraum|\bnische\b|\bnicchia\b"),
    ("balcony", r"balkon|loggia|terras|terrace|parveke|rodu|altan|\bbalcon|sitzplatz|patio|veranda|dachgarten"),
    ("corridor", r"korridor|\bflur\b|\bgang\b|couloir|corridoio|disimpegno|kaytava|koridor|pasillo|\bhall\b|\bdiele\b|"
                 r"\bkorr\b|flur\b|eteinen|\besik\b|vestibule|entree|eingang|ingresso|entrance|lobby|foyer|fuajee|trepp|escalier|"
                 r"scala|treppenhaus|stair|trapp|trepikoda|portaikko|cage d|vano scala|passage|vorraum|vorplatz|\bet\b"),
    ("office", r"buro|buero|office|kontor|toimisto|kabinet|arbeitszimmer|arbeidsrom|arbeitsraum|besprechung|konferenz|"
               r"meeting|sitzung|bureau|ufficio|oficina|reception|empfang|gewerbe|laden|shop|verkauf|kantine|cafeteria|"
               r"klassenzimmer|seminar|auditorium|horsaal|labor\b|\blab\b|andel|kantine|workspace|lounge|veranstaltung|venterum|udstilling|\bmode\b|\bmoede\b|omkl|moderum|puhkeruum|personali"),
]
RULES = [(name, re.compile(rx)) for name, rx in RULES]


def subtype_for(label: str, dict_rows: dict | None = None) -> str | None:
    """Semantic type for a label. None means unknown -> unknown_labels.

    Exact dictionary first (the hand-written seed and anything resolved in a batch), then the word roots above.
    """
    n = norm(label)
    if not n:
        return None
    if dict_rows and n in dict_rows:
        return dict_rows[n]
    if n in SEED:
        return SEED[n]
    # the printed room number or an id in brackets is not part of the name
    t = fold(re.sub(r"\([^)]*\)?", " ", label or ""))
    t = re.sub(r"\b[a-z]?\d+([.\-/]\d+)*\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .,:;-/")
    if not t or AREA_RE.match(label.strip()):
        return None
    for name, rx in RULES:
        if rx.search(t):
            return name
    return None


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
