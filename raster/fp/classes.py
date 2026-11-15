"""Список классов MVP и маппинг имён из CubiCasa5K model.svg.

Порядок CLASSES = id класса в YOLO. Не менять порядок после начала обучения.
"""

CLASSES = [
    # конструктив и проёмы
    "wall", "door", "window",
    # сантехника
    "toilet", "sink", "bathtub", "shower",
    # комнаты
    "room_living", "room_bedroom", "room_kitchen", "room_bath",
    "room_entry", "room_storage", "room_outdoor", "room_other",
    # v2 (дообучение поверх v1): новые классы только в конец, чтобы старые id не сдвигались.
    # Скамья сауны: в CubiCasa5K сауну рисуют скамьями (3235 из 5000 планов), а Space Sauna всего 133
    "sauna_bench",
]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}
ROOM_CLASSES = {c for c in CLASSES if c.startswith("room_")}

# CubiCasa "Space <Name>" -> наш класс комнаты (по мотивам rooms_selected из репозитория CubiCasa5k)
SPACE_MAP = {
    "LivingRoom": "room_living", "Lounge": "room_living", "Dining": "room_living",
    "EatingArea": "room_living",
    "Kitchen": "room_kitchen",
    "Bedroom": "room_bedroom",
    "Bath": "room_bath", "Sauna": "room_bath", "Toilet": "room_bath", "WC": "room_bath",
    "Entry": "room_entry", "HallWay": "room_entry", "Hall": "room_entry", "DraughtLobby": "room_entry",
    "Storage": "room_storage", "Closet": "room_storage", "DressingRoom": "room_storage",
    "Pantry": "room_storage", "Utility": "room_storage",
    "Outdoor": "room_outdoor", "Balcony": "room_outdoor", "Terrace": "room_outdoor",
}
DEFAULT_ROOM = "room_other"
# Space-типы, которые не являются комнатами
SPACE_SKIP = {"Background", "Wall", "Railing", "OpenToBelow", "Below150cm", "Elevated"}

# CubiCasa "FixedFurniture <Name>" -> наш класс (точные имена, сверено с датасетом scripts/inspect_cubicasa.py)
FIXTURE_MAP = {
    "Toilet": "toilet", "Urinal": "toilet",
    "Bathtub": "bathtub", "Jacuzzi": "bathtub",
    "Shower": "shower", "ShowerCab": "shower", "ShowerPlatform": "shower",
    "Sink": "sink", "RoundSink": "sink", "CornerSink": "sink", "DoubleSink": "sink",
    "DoubleSinkRight": "sink", "SideSink": "sink", "WaterTap": "sink",
    "SaunaBench": "sauna_bench", "SaunaBenchLow": "sauna_bench", "SaunaBenchMid": "sauna_bench",
    "SaunaBenchHigh": "sauna_bench",
}
# известные иконки вне MVP — молча пропускаем (ShowerScreen = шторка/стекло, не сам душ)
FIXTURE_SKIP = {"ShowerScreen", "Closet", "CoatCloset", "ElectricalAppliance", "BaseCabinet", "WallCabinet",
                "Housing", "Misc", "Fireplace", "Chimney", "CoatRack", "CounterTop"}

# Синонимы для чата: русские/английские слова -> класс (или группа классов)
SYNONYMS = {
    "wall": ["стен", "wall"],
    "door": ["двер", "door", "проём", "проем"],
    "window": ["окн", "окон", "window"],
    "toilet": ["туалет", "унитаз", "toilet", "wc"],
    "sink": ["раковин", "умывальник", "мойк", "sink"],
    "bathtub": ["ванн", "bathtub", "bath tub"],
    "shower": ["душ", "shower"],
    "room_bedroom": ["спальн", "спален", "bedroom"],
    "room_kitchen": ["кухн", "кухонь", "kitchen"],
    "room_living": ["гостин", "living"],
    "room_bath": ["санузл", "санузел", "ванная комната", "ванные комнаты", "bathroom"],
    "room_entry": ["прихож", "коридор", "холл", "entry", "hall"],
    "room_storage": ["кладов", "гардероб", "storage", "closet"],
    "room_outdoor": ["балкон", "террас", "outdoor", "balcon", "parveke", "terrace"],
    # виртуальные классы (ответ — из описи плана, fp/inventory.py; YOLO — для сверки, где есть)
    "shower_bathtub": ["души и ванн", "душей и ванн", "душевых и ванн", "showers and bathtubs",
                       "showers and baths", "shower and bathtub", "showers or bathtubs"],
    "wc_room": ["wc room", "wc-room", "toilet room", "туалетных комнат", "туалетные комнат", "отдельных туалет",
                "отдельные туалет", "отдельных wc"],
    "floor": ["floors", "storeys", "storey", "stories", "этаж"],
    "sauna": ["саун", "sauna", "bastu"],          # виртуальный: помещения со скамьями сауны
    "sauna_bench": ["скамь", "полок"],
    "room": ["комнат", "помещен", "room"],
}

LABELS_RU = {
    "wall": "стены", "door": "двери", "window": "окна", "toilet": "унитазы",
    "sink": "раковины", "bathtub": "ванны", "shower": "душевые",
    "room_living": "гостиные", "room_bedroom": "спальни", "room_kitchen": "кухни",
    "room_bath": "санузлы", "room_entry": "прихожие/коридоры", "room_storage": "кладовые",
    "room_outdoor": "балконы/террасы", "room_other": "прочие помещения", "sauna_bench": "скамьи сауны", "sauna": "сауны", "zone": "зоны легенды", "space": "помещения (экспликация)",
    "shower_bathtub": "души и ванны", "wc_room": "туалетные комнаты (WC)", "floor": "этажи",
    "pipe": "трубы", "fitting": "арматура и приборы",
}

# Цвета оверлея (и для визуальной проверки, и для фронта)
COLORS = {
    "wall": "#333333", "door": "#c0502b", "window": "#3a78b8", "toilet": "#8e44ad",
    "sink": "#16a085", "bathtub": "#2980b9", "shower": "#27ae60",
    "room_living": "#f5d76e", "room_bedroom": "#a3d9a5", "room_kitchen": "#f7b267",
    "room_bath": "#9ad0ec", "room_entry": "#d7bde2", "room_storage": "#cccccc",
    "room_outdoor": "#b5e7a0", "room_other": "#eeeeee", "sauna_bench": "#e59866", "zone": "#7048e8", "space": "#1c7ed6",
    "pipe": "#e8590c", "fitting": "#2b8a3e",
}
