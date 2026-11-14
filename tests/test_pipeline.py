import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fp import infer  # noqa: E402
from fp.chat import chat, parse_rule  # noqa: E402
from fp.convert_yolo import convert_one  # noqa: E402
from fp.cubicasa_svg import parse_cubicasa_svg  # noqa: E402
from fp.query import count, highlight, level  # noqa: E402
from fp.scene import Element, Scene  # noqa: E402


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic.py"), "--out", str(out), "--n", "2"], check=True)
    return out / "synthetic" / "0000"


@pytest.fixture(scope="session")
def scene(synth, tmp_path_factory):
    return infer.run(synth / "F1_scaled.png", tmp_path_factory.mktemp("scene"), gt_svg=synth / "model.svg")


def test_parse_svg(synth):
    p = parse_cubicasa_svg(synth / "model.svg")
    classes = {c for c, _ in p.objects}
    assert {"wall", "door", "window", "toilet", "sink", "room_bath", "room_kitchen"} <= classes
    assert not p.unknown  # Closet — в FIXTURE_SKIP


def test_convert_normalized(synth):
    lines, _, (w, h) = convert_one(synth / "model.svg", synth / "F1_scaled.png")
    assert lines
    for ln in lines:
        vals = [float(v) for v in ln.split()[1:]]
        assert len(vals) % 2 == 0 and len(vals) >= 6
        assert all(0 <= v <= 1 for v in vals)


def test_scene_rooms_assigned(scene):
    toilets = [o for o in scene.objects if o.cls == "toilet"]
    assert len(toilets) == 1 and toilets[0].room == "bath_1"
    kitchen_sinks = [o for o in scene.objects if o.cls == "sink" and o.room == "kitchen_1"]
    assert len(kitchen_sinks) == 1


def test_scene_roundtrip(scene, tmp_path):
    scene.save(tmp_path)
    again = Scene.load(tmp_path)
    assert again.to_dict() == scene.to_dict()
    svg = (tmp_path / "scene.svg").read_text()
    assert 'class="toilet"' in svg and 'data-room="bath_1"' in svg


@pytest.mark.parametrize("msg,expected", [
    ("выдели все туалеты", ("highlight", {"cls": "toilet"})),
    ("Покажи окна", ("highlight", {"cls": "window"})),
    ("сколько раковин на кухне", ("count", {"cls": "sink", "room": "kitchen"})),
    ("сколько дверей", ("count", {"cls": "door"})),
    ("какие комнаты есть", ("list_rooms", {})),
    ("площадь кухни", ("area", {"room": "kitchen"})),
    ("что в ванной", ("objects_in", {"room": "bath"})),
    ("очисти", ("clear", {})),
    ("выдели ванны", ("highlight", {"cls": "bathtub"})),
    ("покажи санузлы", ("highlight", {"cls": "room_bath"})),
])
def test_rule_parser(msg, expected):
    assert parse_rule(msg) == expected


def test_chat_highlight(scene):
    r = chat(scene, "выдели все раковины", use_llm=False)
    assert r["parsed_by"] == "cache" and r["action"]["type"] == "highlight"
    assert len(r["action"]["ids"]) == 2 and r["confidence"] == "high"


def test_chat_unknown_without_llm(scene):
    r = chat(scene, "расскажи анекдот", use_llm=False)
    assert r["parsed_by"] == "none" and r["action"]["type"] == "none"


# --- правила confidence.md ---
def _scene(confs, cls="toilet", secondary=0):
    objs = [Element(f"t{i}", cls, [[0, 0], [1, 0], [1, 1]], c) for i, c in enumerate(confs)]
    objs += [Element(f"o{i}", cls, [[0, 0], [1, 0], [1, 1]], 0.9, "ocr", primary=False) for i in range(secondary)]
    return Scene("x.png", 10, 10, objs, [])


def test_conf_min():
    assert count(_scene([0.95, 0.75]), "toilet").confidence == "medium"


def test_conf_zero_is_low():
    a = count(_scene([]), "toilet")
    assert a.answer == 0 and a.confidence == "low"


def test_conf_two_sources_agree():
    assert count(_scene([0.6, 0.6], secondary=2), "toilet").confidence == "high"


def test_conf_two_sources_disagree():
    a = count(_scene([0.99, 0.99], secondary=1), "toilet")
    assert a.confidence == "low" and "разошлись" in a.note


def test_levels():
    assert (level(0.9), level(0.7), level(0.69)) == ("high", "medium", "low")


def test_highlight_bad_class():
    with pytest.raises(ValueError):
        highlight(_scene([]), "sofa")


def test_preprocess_runs_on_skewed(synth):
    import cv2
    import numpy as np
    from fp.preprocess import preprocess
    img = cv2.imread(str(synth / "F1_scaled.png"))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
    skewed = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    out, info = preprocess(skewed)
    assert abs(abs(info["deskew_deg"]) - 3.0) < 0.6
    assert out.ndim == 3 and max(out.shape[:2]) <= 2048


def test_fixture_transform_and_rect_fallback():
    svg = b'''<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50"><g transform="matrix(1,0,0,1,0,0)">
    <g class="FixedFurniture Toilet" transform="matrix(1,0,0,1,100,200)"><g class="BoundaryPolygon"><polygon points="0,0 40,0 40,70 0,70"/></g></g>
    <g class="FixedFurniture Toilet" transform="matrix(-1,0,0,1,500,10)"><g class="BoundaryPolygon"><rect x="0" y="0" width="40" height="18"/></g><desc>Width:40 Height:85 Depth:70 Elevation:0</desc></g>
    <g class="FixedFurniture ShowerScreen" transform="translate(5,5)"><g class="BoundaryPolygon"><polygon points="0,0 1,0 1,1"/></g></g>
    </g></svg>'''
    p = parse_cubicasa_svg(svg)
    assert [c for c, _ in p.objects] == ["toilet", "toilet"]
    assert p.objects[0][1][0] == (100, 200) and p.objects[0][1][2] == (140, 270)
    xs = [x for x, _ in p.objects[1][1]]
    assert min(xs) == 460 and max(xs) == 500  # отражение по X + сдвиг, габарит из desc


def test_sauna_counts_rooms_with_benches():
    from fp.scene import Scene
    from fp import query
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    dets = [("room_bath", sq(0, 0, 100), 0.95), ("room_bath", sq(200, 0, 100), 0.95),
            ("sauna_bench", sq(10, 10, 20), 0.95), ("sauna_bench", sq(40, 10, 20), 0.95),
            ("sauna_bench", sq(210, 10, 20), 0.95)]
    sc = Scene.build("x.png", 400, 200, dets, source="yolo")
    assert query.count(sc, "sauna").answer == 2
    assert query.count(sc, "sauna_bench").answer == 3
    from fp.chat import parse_rule
    assert parse_rule("сколько саун?")[1]["cls"] == "sauna"


def test_merge_duplicates_joins_split_window():
    import numpy as np
    from fp.infer import merge_duplicates
    a = (np.array([[0, 0], [60, 0], [60, 10], [0, 10]]), 0.6, [0, 0, 60, 10])
    b = (np.array([[50, 0], [110, 0], [110, 10], [50, 10]]), 0.8, [50, 0, 110, 10])
    c = (np.array([[300, 0], [360, 0], [360, 10], [300, 10]]), 0.9, [300, 0, 360, 10])
    out = merge_duplicates([a, b, c], 0.1)
    assert len(out) == 2
    assert max(o[1] for o in out if o[2][0] == 0) == 0.8


def test_sauna_loose_benches_cluster():
    from fp.scene import Scene
    from fp import query
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    dets = [("sauna_bench", sq(500, 500, 20), 0.95), ("sauna_bench", sq(525, 500, 20), 0.95),
            ("sauna_bench", sq(900, 900, 20), 0.95)]
    sc = Scene.build("x.png", 1200, 1200, dets, source="yolo")
    assert query.count(sc, "sauna").answer == 2


def test_calibrated_confidence_uses_borderline_detections():
    from fp.scene import Scene
    from fp import query
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    t = query.CAL["toilet"]["conf"]
    sure = Scene.build("x", 500, 500, [("toilet", sq(10, 10, 20), 0.97)], meta={"candidates": []})
    assert query.count(sure, "toilet").confidence == "high"
    shaky = Scene.build("x", 500, 500, [("toilet", sq(10, 10, 20), 0.97)],
                        meta={"candidates": [["toilet", t - 0.05, 200, 200], ["toilet", t - 0.02, 300, 300]]})
    assert query.count(shaky, "toilet").confidence == "low"


def test_plural_genitive_room_words():
    from fp.chat import parse_rule
    assert parse_rule("сколько спален?")[1]["cls"] == "room_bedroom"
    assert parse_rule("сколько кухонь")[1]["cls"] == "room_kitchen"


def test_llm_path_with_fake_client(monkeypatch):
    """Цикл tool use: модель вызывает highlight, получает результат, отвечает текстом; подсветка доходит до фронта."""
    from types import SimpleNamespace as NS
    import anthropic
    from fp import chat as chat_mod
    from fp.scene import Scene
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    sc = Scene.build("x", 500, 500, [("window", sq(10, 10, 20), 0.97), ("window", sq(100, 10, 20), 0.97)],
                     meta={"candidates": []})
    seen = []

    class FakeMessages:
        def create(self, **kw):
            seen.append(kw)
            usage = NS(input_tokens=1000, output_tokens=50)
            if len(seen) == 1:
                return NS(stop_reason="tool_use", usage=usage, content=[
                    NS(type="tool_use", id="tu1", name="highlight", input={"cls": "window"})])
            return NS(stop_reason="end_turn", usage=usage, content=[NS(type="text", text="Found 2 windows.")])

    monkeypatch.setattr(anthropic, "Anthropic", lambda: NS(messages=FakeMessages(), beta=NS(messages=FakeMessages())))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    out = chat_mod.chat(sc, "Mark every window please", [])
    assert out["parsed_by"] == "llm" and out["reply"] == "Found 2 windows."
    assert out["action"]["type"] == "highlight" and len(out["action"]["ids"]) == 2
    assert seen[0]["model"] == "claude-haiku-5-5" and seen[0]["thinking"] == {"type": "disabled"}
    assert out["cost_usd"] == round((2000 * 0.10 + 100 * 0.50) / 1e6, 6)


def test_show_and_count_highlights():
    from fp.chat import parse_rule
    assert parse_rule("покажи где спальни и сколько их")[0] == "highlight"
    assert parse_rule("сколько спален")[0] == "count"


def test_scale_from_labels_and_area_m2():
    from fp.scene import Scene
    from fp import query, scale
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    # 50 px/м: комнаты 200x200 px = 16 м², 100x100 px = 4 м²
    sc = Scene.build("x", 600, 600, [("room_kitchen", sq(0, 0, 200), 0.95), ("room_bath", sq(300, 0, 100), 0.95),
                                     ("room_bedroom", sq(0, 300, 250), 0.95)])
    labels = [{"value_m2": 16.0, "x": 100, "y": 100}, {"value_m2": 4.0, "x": 350, "y": 50},
              {"value_m2": 25.0, "x": 125, "y": 425}, {"value_m2": 99.0, "x": 590, "y": 590}]  # последняя вне комнат
    est = scale.from_area_labels(sc, labels)
    assert abs(est["px_per_m"] - 50) < 0.5 and est["confidence"] == "high"
    sc.meta["scale"], sc.meta["px_per_m"] = est, est["px_per_m"]
    a = query.area(sc, "kitchen")
    assert a.unit == "m2" and a.answer[0]["area_m2"] == 16.0 and a.answer[0]["label_m2"] == 16.0
    assert a.confidence == "high"


def test_scale_two_points_feet():
    from fp import scale
    est = scale.from_two_points((0, 0), (300, 0), 80, "ft")
    assert abs(est["px_per_m"] - 300 / (80 * 0.3048)) < 1e-6


def test_measure_bar():
    import numpy as np
    from fp import scale
    img = np.full((100, 400, 3), 255, np.uint8)
    img[50:56, 40:340] = 0          # шкала 300 px
    img[50:56, 100:110] = 255       # белый сегмент шкалы (чередование) — короче разрыва не считается
    assert scale.measure_bar(img, [30, 40, 350, 60]) == 300.0


def test_find_zones_by_legend_color_and_legend_answer():
    import numpy as np
    from fp import sheet, query
    from fp.scene import Scene, Element
    img = np.full((400, 600, 3), 255, np.uint8)
    img[50:150, 50:250] = (180, 210, 190)      # зона «DORMITORY» (BGR)
    img[200:300, 50:150] = (120, 130, 220)     # зона «COMMON AREA»
    img[20:30, 500:530] = (180, 210, 190)      # образец легенды — должен быть исключён
    legend = [{"label": "DORMITORY", "color_hex": "#bed2b4", "swatch_bbox": [500, 20, 530, 30]},
              {"label": "COMMON AREA", "color_hex": "#dc8278", "swatch_bbox": None}]
    zones = sheet.find_zones(img, legend, exclude=[[500, 20, 530, 30]])
    assert sorted(z["label"] for z in zones) == ["COMMON AREA", "DORMITORY"]
    sc = Scene.build("x", 600, 400, [])
    sc.meta.update({"px_per_m": 10.0, "sheet": {"legend": legend, "tables": [{"title": "SQUARE FOOTAGE", "rows": [
        {"label": "WEST WING DORMITORY", "value": 2153, "unit": "SF", "text": "WEST WING DORMITORY = 2,153 SF"}]}]},
        "zones": {}})
    for n, z in enumerate(zones, 1):
        sc.objects.append(Element(f"zone{n}", "zone", z["polygon"], 0.8, source="legend"))
        sc.meta["zones"][f"zone{n}"] = {"label": z["label"], "color": z["color"]}
    a = query.legend(sc, "dormitory")
    assert a.answer["legend_label"] == "DORMITORY" and a.answer["zones_on_plan"] == 1
    assert abs(a.answer["color_area_m2"] - 200.0) < 6     # 200x100 px при 10 px/м
    assert a.confidence == "high"                          # 2153 SF = 200 м² — совпало с таблицей
    assert query.legend(sc).answer["legend"][0]["zones_on_plan"] == 1


def test_legend_label_matching_prefers_specific():
    from fp import query
    from fp.scene import Scene
    sc = Scene.build("x", 100, 100, [])
    sc.meta.update({"sheet": {"legend": [{"label": "DORMITORY"}, {"label": "ACCESSIBLE (ABA) DORMITORY"}], "tables": []},
                    "zones": {}})
    assert query.legend(sc, "accessible dormitory").answer["legend_label"] == "ACCESSIBLE (ABA) DORMITORY"
    assert query.legend(sc, "dormitory").answer["legend_label"] == "DORMITORY"


def test_spaces_from_labels_one_row_per_label_and_area():
    import numpy as np
    from fp.rooms_cv import spaces_from_labels
    img = np.full((300, 500, 3), 255, np.uint8)
    img[50:54, 50:450] = 0; img[246:250, 50:450] = 0; img[50:250, 50:54] = 0; img[50:250, 446:450] = 0
    img[50:250, 248:252] = 0                                   # перегородка: две комнаты 200x200 px
    labels = [{"text": "A", "x": 150, "y": 150, "value_m2": 4.0}, {"text": "A2", "x": 160, "y": 150},
              {"text": "B", "x": 350, "y": 150}]
    res = spaces_from_labels(img, labels, px_per_m=100.0)
    assert len(res) == 3 and res[0]["ok"] and not res[1]["ok"] and res[2]["ok"]
    assert abs(res[0]["area_m2"] - 3.8) < 0.25                 # внутри стен ~194x194 px при 100 px/м


def test_fixture_second_opinion_lowers_confidence():
    from fp.scene import Scene
    from fp import query
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    sc = Scene.build("x", 500, 500, [("toilet", sq(10, 10, 20), 0.97)], meta={"candidates": []})
    assert query.count(sc, "toilet").confidence == "high"
    sc.meta["fixture_check"] = {"done": True, "counts": {"toilet": 0, "sink": 1}}
    a = query.count(sc, "toilet")
    assert a.answer == 1 and a.confidence == "medium" and "второе мнение" in a.note
    sc.meta["fixture_check"]["counts"]["toilet"] = 1
    assert query.count(sc, "toilet").confidence == "high"


def test_drop_detections_outside_drawing():
    from fp.infer import drop_outside_drawing
    from fp.scene import Scene
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    sc = Scene.build("x", 1000, 600, [("room_other", sq(100, 100, 200), 0.6), ("room_other", sq(900, 50, 80), 0.6),
                                       ("door", sq(150, 150, 20), 0.6), ("door", sq(250, 150, 20), 0.6),
                                       ("window", sq(350, 150, 20), 0.6)])
    assert drop_outside_drawing(sc, [0, 0, 800, 600]) == 1          # 1 из 5 — графа штампа справа, убираем
    assert [r.id for r in sc.rooms] == ["other_1"] and len(sc.objects) == 3


def test_drop_outside_drawing_skips_partial_frame():
    from fp.infer import drop_outside_drawing
    from fp.scene import Scene
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    sc = Scene.build("x", 1000, 600, [("room_other", sq(100, 100, 100), 0.6), ("room_other", sq(600, 100, 100), 0.6),
                                       ("room_other", sq(800, 100, 100), 0.6)])
    assert drop_outside_drawing(sc, [0, 0, 400, 600]) == 0 and len(sc.rooms) == 3   # рамка вокруг одного этажа
