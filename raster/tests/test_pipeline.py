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


def test_display_copy_is_sharper_with_same_crop(tmp_path):
    """Школа: исходник 14000 px, разбор на 2048 px — для показа отдельная копия крупнее, с той же обрезкой."""
    import cv2
    import numpy as np
    from fp.infer import save_display
    from fp.preprocess import preprocess
    img = np.full((3000, 6000, 3), 255, np.uint8)
    cv2.rectangle(img, (500, 400), (5500, 2600), (0, 0, 0), 6)
    cv2.putText(img, "Group Room", (2000, 1500), 0, 2.0, (0, 0, 0), 3)
    keep: dict = {}
    out, info = preprocess(img, keep=keep)
    hd = keep["display"]
    assert max(out.shape[:2]) == 2048 and hd.shape[1] > 2 * out.shape[1]
    assert abs(hd.shape[1] / hd.shape[0] - out.shape[1] / out.shape[0]) < 0.01      # та же обрезка
    save_display(hd, tmp_path)
    saved = cv2.imread(str(tmp_path / "image_hd.png"), cv2.IMREAD_UNCHANGED)
    assert saved.ndim == 2 and saved.shape == hd.shape[:2]                          # без цвета — серый


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


def test_measure_bar_ignores_frame_line_and_labels():
    """S-100F: над шкалой проходит граница штампа длиннее рамки VLM, над сегментами — подписи 0 500 1000 2000."""
    import numpy as np
    from fp import scale
    img = np.full((120, 500, 3), 255, np.uint8)
    img[30:32, :] = 0                # линия штампа через весь лист
    img[38:44, 150:155] = 0          # подпись «0»
    img[38:44, 240:262] = 0          # подпись «2000» выступает за конец шкалы
    img[48:51, 150:200] = 0          # сегменты-«ступеньки» на разной высоте
    img[51:54, 200:250] = 0
    assert scale.measure_bar(img, [150, 45, 250, 55]) == 100.0


def test_measure_bar_ignores_line_from_one_edge():
    """M-101A: над шкалой линия заходит в рамку слева и обрывается внутри — это не шкала."""
    import numpy as np
    from fp import scale
    img = np.full((120, 500, 3), 255, np.uint8)
    img[30:32, 0:270] = 0            # линия от края листа, обрывается за концом шкалы
    img[48:51, 150:200] = 0
    img[51:54, 200:250] = 0
    assert scale.measure_bar(img, [150, 40, 255, 55]) == 100.0


def test_scale_sources_disagree():
    from fp import scale
    bar = {"px_per_m": 79.0, "source": "scale_bar", "confidence": "medium", "detail": "шкала"}
    text = {"px_per_m": 48.7, "source": "scale_text", "confidence": "low", "detail": "1:50 на A1"}
    dims = {"px_per_m": 48.8, "source": "dimensions", "confidence": "low", "detail": "габарит"}
    # третий источник (габарит) сходится с текстом — берём текст, а не шкалу
    r = scale.choose([bar, text], dims)
    assert r["source"] == "scale_text" and r["confidence"] == "medium" and "разошлись" in r["detail"]
    # третьего нет — шкала, но low и объяснение
    r = scale.choose([bar, text])
    assert r["source"] == "scale_bar" and r["confidence"] == "low" and "разошлись" in r["detail"]
    # совпали в пределах 5% — high, как раньше
    r = scale.choose([{**bar, "px_per_m": 49.0}, text], dims)
    assert r["source"] == "scale_bar" and r["confidence"] == "high"
    # габарит сам по себе не выбирается
    assert scale.choose([bar], dims)["source"] == "scale_bar"


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
    assert [r.id for r in sc.rooms] == ["other_2"] and len(sc.objects) == 3   # id по положению: штамп выше


def test_drop_outside_drawing_skips_partial_frame():
    from fp.infer import drop_outside_drawing
    from fp.scene import Scene
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    sc = Scene.build("x", 1000, 600, [("room_other", sq(100, 100, 100), 0.6), ("room_other", sq(600, 100, 100), 0.6),
                                       ("room_other", sq(800, 100, 100), 0.6)])
    assert drop_outside_drawing(sc, [0, 0, 400, 600]) == 0 and len(sc.rooms) == 3   # рамка вокруг одного этажа


def test_ids_follow_reading_order_not_detection_order():
    from fp.scene import Scene
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    dets = [("door", sq(300, 300, 20), 0.99), ("door", sq(10, 302, 20), 0.5), ("door", sq(200, 10, 20), 0.7),
            ("window", sq(50, 50, 20), 0.8)]
    a = Scene.build("x", 500, 500, dets)
    b = Scene.build("x", 500, 500, list(reversed(dets)))
    pos = lambda sc: {o.id: tuple(o.polygon[0]) for o in sc.objects}
    assert pos(a) == pos(b)
    # d1 — верхняя дверь; d2, d3 — в одной полосе по y (разница 2 px), слева направо
    assert pos(a)["d1"] == (200, 10) and pos(a)["d2"] == (10, 302) and pos(a)["d3"] == (300, 300)


def _fake_llm(monkeypatch, tool_input, text="Found them."):
    from types import SimpleNamespace as NS
    import anthropic
    seen = []

    class FakeMessages:
        def create(self, **kw):
            seen.append(kw)
            usage = NS(input_tokens=1000, output_tokens=50)
            if len(seen) % 2 == 1:
                return NS(stop_reason="tool_use", usage=usage, content=[
                    NS(type="tool_use", id=f"tu{len(seen)}", name=tool_input[0], input=tool_input[1])])
            return NS(stop_reason="end_turn", usage=usage, content=[NS(type="text", text=text)])

    monkeypatch.setattr(anthropic, "Anthropic", lambda: NS(messages=FakeMessages(), beta=NS(messages=FakeMessages())))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    return seen


def test_answer_and_template_cache(monkeypatch, tmp_path):
    """Повтор вопроса к тому же плану — из кэша ответов; тот же вопрос к другому плану — шаблон без LLM;
    правка сцены (масштаб) делает старый ответ недействительным."""
    from fp import cache
    from fp import chat as chat_mod
    from fp.scene import Scene
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    cache.configure(tmp_path / "c.sqlite")
    try:
        seen = _fake_llm(monkeypatch, ("highlight", {"cls": "window"}))
        a = Scene.build("x", 500, 500, [("window", sq(10, 10, 20), 0.97), ("window", sq(100, 10, 20), 0.97)])
        b = Scene.build("x", 500, 500, [("window", sq(10, 10, 20), 0.97)] * 1 + [("window", sq(200, 300, 20), 0.97),
                                                                                  ("window", sq(300, 300, 20), 0.97)])
        q = "Mark every window please"
        first = chat_mod.chat(a, q, [])
        assert first["parsed_by"] == "llm" and first["cost_usd"] > 0 and len(seen) == 2
        again = chat_mod.chat(a, "  mark every window please? ", [])
        assert again["parsed_by"] == "answer_cache" and again["cost_usd"] == 0 and again["reply"] == first["reply"]
        assert again["saved_usd"] == first["cost_usd"] and len(seen) == 2
        other = chat_mod.chat(b, q, [])
        assert other["parsed_by"] == "template" and other["cost_usd"] == 0 and len(seen) == 2
        assert other["action"]["ids"] == ["win1", "win2", "win3"] and "3" in other["reply"]
        assert other["reply"].startswith("Highlighted window")       # английский вопрос — английский ответ
        a.meta["px_per_m"] = 50                                      # сцена изменилась
        assert chat_mod.chat(a, q, [])["parsed_by"] == "template"
        # уточнение в диалоге («а в спальне?») зависит от истории — кэшу не доверяем
        n = len(seen)
        hist = [{"role": "user", "content": q}, {"role": "assistant", "content": "ok"}]
        for _ in range(2):
            assert chat_mod.chat(a, "and in the bedroom", hist)["parsed_by"] == "llm"
        assert len(seen) == n + 4
        # самостоятельный вопрос посреди диалога — тоже бесплатно (сцена после масштаба новая -> шаблон)
        assert chat_mod.chat(a, q, hist)["parsed_by"] == "template" and len(seen) == n + 4
        st = cache.stats()
        assert st["template_hits"] == 3 and st["templates"] == 1
    finally:
        cache.configure(None)


def test_template_not_stored_for_plan_specific_ids(monkeypatch, tmp_path):
    from fp import cache
    from fp import chat as chat_mod
    from fp.scene import Scene
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    cache.configure(tmp_path / "c.sqlite")
    try:
        _fake_llm(monkeypatch, ("objects_in", {"room": "bedroom_2"}))
        sc = Scene.build("x", 500, 500, [("room_bedroom", sq(0, 0, 100), 0.9), ("room_bedroom", sq(200, 0, 100), 0.9)])
        chat_mod.chat(sc, "what is inside the second bedroom", [])
        assert cache.stats()["templates"] == 0 and cache.stats()["answers"] == 1
    finally:
        cache.configure(None)


def test_rule_reply_in_english():
    from fp.chat import chat
    from fp.scene import Scene
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    sc = Scene.build("x", 500, 500, [("door", sq(10, 10, 20), 0.97)])
    out = chat(sc, "how many doors", use_llm=False)
    assert out["parsed_by"] == "cache" and out["reply"].startswith("Found door: 1")


def _walls_sheet():
    """Лист 1000×700: комната A (чистая) и зал B, забитый «трубами» (сетка линий), стены — тонкие двойные линии."""
    import cv2
    import numpy as np
    img = np.full((700, 1000, 3), 255, np.uint8)
    for d in (0, 4):  # двойные линии стен
        cv2.rectangle(img, (100 - d, 100 - d), (900 + d, 600 + d), (150, 150, 150), 1)
        cv2.line(img, (300 + d, 100), (300 + d, 600), (150, 150, 150), 1)
    for x in range(340, 900, 60):  # трубы в зале B — дробят пол на ячейки
        cv2.line(img, (x, 100), (x, 600), (0, 0, 0), 1)
    for y in range(150, 600, 70):
        cv2.line(img, (300, y), (900, y), (0, 0, 0), 1)
    cv2.line(img, (320, 130), (880, 130), (0, 0, 0), 2)  # длинная труба вдоль верхней стены
    return img


def test_rooms_from_bbox_snaps_to_walls_and_fills_clean_room():
    import cv2
    from fp.rooms_cv import rooms_from_bbox
    g = cv2.cvtColor(_walls_sheet(), cv2.COLOR_BGR2GRAY)
    a = rooms_from_bbox(g, [120, 115, 285, 580], 50.0)            # рамка VLM чуть внутри комнаты A
    assert len(a) == 1 and a[0]["how"] == "fill"
    assert abs(a[0]["area_px2"] - 199 * 499) / (199 * 499) < 0.03
    b = rooms_from_bbox(g, [318, 112, 893, 590], 50.0)           # зал с трубами: заливка дробится -> рамка
    assert len(b) == 1 and b[0]["how"] == "bbox" and b[0]["sides_snapped"] == 4
    xs = [p[0] for p in b[0]["polygon"]]
    ys = [p[1] for p in b[0]["polygon"]]
    # прижато к стенам (305..899 × 101..599), а не к трубе вдоль верхней стены (y=130)
    assert abs(min(xs) - 305) <= 3 and abs(max(xs) - 899) <= 3 and abs(min(ys) - 101) <= 3 and abs(max(ys) - 599) <= 3


def test_rooms_by_walls_replaces_model_rooms(monkeypatch):
    from fp import infer, sheet
    from fp.query import call
    from fp.scene import Scene
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    sc = Scene.build("x", 1000, 700, [("room_outdoor", sq(920, 620, 40), 0.3), ("toilet", sq(150, 150, 30), 0.9)],
                     meta={"px_per_m": 50.0})
    monkeypatch.setattr(sheet, "find_rooms", lambda img, region=None: (
        [{"name": "WC", "named": True, "bbox": [120, 115, 285, 580]},
         {"name": "pump room", "named": False, "bbox": [318, 112, 893, 590]}], 0.0003))
    rows = infer.rooms_by_walls(_walls_sheet(), sc, {"drawing_bbox": None})
    assert [r["id"] for r in rows] == ["bath_1", "other_1"]
    assert sc.meta["yolo_rooms_replaced"] == ["outdoor_1"] and [r.id for r in sc.rooms] == ["bath_1", "other_1"]
    assert [o.room for o in sc.objects if o.cls == "toilet"] == ["bath_1"]
    from fp.spaces import build
    build(_walls_sheet(), sc)
    a = call(sc, "rooms", {})
    assert [r["name"] for r in a.answer["rooms"]] == ["WC", "pump room"]
    assert abs(a.answer["rooms"][1]["area_m2"] - 594 * 498 / 2500) < 1.5 and a.confidence == "medium"


def test_area_labels_two_far_apart_estimates_do_not_crash():
    from fp.scale import from_area_labels
    from fp.scene import Scene
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    sc = Scene.build("x", 800, 800, [("room_kitchen", sq(0, 0, 100), 0.9), ("room_bath", sq(300, 0, 300), 0.9)])
    r = from_area_labels(sc, [{"value_m2": 10, "x": 50, "y": 50}, {"value_m2": 10, "x": 450, "y": 150}])
    assert r is not None and r["px_per_m"] > 0 and r["confidence"] == "low"


@pytest.mark.parametrize("msg,expected", [
    ("How many sinks and wash basins (kitchen and bathroom combined; a double-bowl sink counts as one) are drawn "
     "on this floor plan?", ("count", {"cls": "sink"})),
    ("How many showers and bathtubs are drawn on this floor plan in total?", ("highlight", {"cls": "shower_bathtub"})),
    ("How many separate WC rooms (toilet rooms marked WC) are on this floor plan?", ("count", {"cls": "wc_room"})),
    ("How many floors (storeys) does this floor plan show?", ("count", {"cls": "floor"})),
    ("How many balconies (parveke, including glazed balconies) are on this floor plan?",
     ("count", {"cls": "room_outdoor"})),
    ("How many bathrooms or washrooms (e.g. KH, KPH, PH, PESUH, SH) are on this floor plan, not counting separate "
     "WC rooms?", ("count", {"cls": "room_bath"})),
    ("How many toilet bowls (WC seats) are drawn on this floor plan?", ("count", {"cls": "toilet"})),
    ("сколько этажей", ("count", {"cls": "floor"})),
])
def test_rule_parser_stoa_questions(msg, expected):
    assert parse_rule(msg) == expected


def test_count_routes_between_model_and_inventory(monkeypatch):
    from fp import query
    from fp.scene import Scene
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    sc = Scene.build("x", 500, 500, [("sink", sq(10, 10, 20), 0.97), ("toilet", sq(100, 10, 20), 0.97)])
    sc.meta["inventory"] = {"sinks": 2, "toilets": 1, "floors": 2, "wc_rooms": 1}
    monkeypatch.setattr(query, "ROUTE", {"sink": "inventory", "toilet": "yolo"})
    s = query.count(sc, "sink")                     # отвечает опись, подсвечено распознанное, разошлись на 1
    assert s.answer == 2 and s.ids == ["s1"] and s.confidence == "medium"
    t = query.count(sc, "toilet")                   # отвечает модель, опись согласна -> high
    assert t.answer == 1 and t.confidence == "high"
    assert query.count(sc, "floor").answer == 2     # классов нет в YOLO — только опись
    assert query.count(sc, "sink", room="kitchen").answer == 0   # по комнате — только модель
    sc.meta.pop("inventory")
    assert query.count(sc, "floor").answer is None and query.count(sc, "sink").answer == 1


def _door_sheet():
    """Лист 600×400, 50 px/м: стена с проёмом 0.9 м, створка + четверть дуги; рядом полный круг (марка) и текст."""
    import cv2
    import numpy as np
    img = np.full((400, 600, 3), 255, np.uint8)
    cv2.line(img, (50, 200), (250, 200), (150, 150, 150), 2)          # стена слева от проёма
    cv2.line(img, (295, 200), (550, 200), (150, 150, 150), 2)         # стена справа (проём 250..295 = 0.9 м)
    cv2.line(img, (250, 200), (250, 155), (190, 190, 190), 1)         # створка от петли (250,200) вверх
    cv2.ellipse(img, (250, 200), (45, 45), 0, -90, 0, (220, 220, 220), 1)   # дуга открывания, бледная
    cv2.circle(img, (430, 300), 40, (0, 0, 0), 1)                    # марка — полный круг, не дверь
    cv2.putText(img, "CCC OOO", (60, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return img


def test_find_door_arcs_finds_faint_swing_not_circles_or_text():
    import cv2
    from fp.doors_cv import find_door_arcs
    arcs = find_door_arcs(cv2.cvtColor(_door_sheet(), cv2.COLOR_BGR2GRAY), 50.0)
    assert len(arcs) == 1
    hx, hy = arcs[0]["hinge"]
    assert abs(hx - 250) <= 3 and abs(hy - 200) <= 3 and abs(arcs[0]["r_m"] - 0.9) <= 0.1


def test_arc_doors_join_model_doors():
    from fp.infer import add_arc_doors
    from fp.query import count
    from fp.scene import Scene
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    wall = [(50, 197), (550, 197), (550, 203), (50, 203)]
    sc = Scene.build("x", 600, 400, [("wall", wall, 0.9), ("door", sq(60, 190, 20), 0.4)], meta={"px_per_m": 50.0})
    info = add_arc_doors(_door_sheet(), sc)
    assert info["added"] == 1 and info["confirmed"] == []
    sc2 = Scene.build("x", 600, 400, [("door", sq(60, 190, 20), 0.4)], meta={"px_per_m": 50.0})
    sc2.rooms = []
    assert add_arc_doors(_door_sheet(), sc2)["added"] == 1     # нет ни стен, ни помещений — фильтр не применяется
    a = count(sc, "door")
    assert a.answer == 2 and a.confidence == "medium" and "дугам" in a.note


def _window_sheet():
    """Лист 600×300, 50 px/м: толстая стена с простенками, между ними три тонкие линии остекления (окна 1 м);
    внутри — стол из двух параллельных линий, не упирающийся в стену."""
    import cv2
    import numpy as np
    img = np.full((300, 600, 3), 255, np.uint8)
    for x0, x1 in ((40, 100), (150, 200), (250, 300), (350, 560)):    # стена и простенки, толщина 12 px
        cv2.rectangle(img, (x0, 100), (x1, 111), (0, 0, 0), -1)
    for x0, x1 in ((100, 150), (200, 250), (300, 350)):                 # остекление 3 линии
        for y in (101, 105, 110):
            cv2.line(img, (x0, y), (x1, y), (215, 215, 215), 1)
    cv2.line(img, (200, 200), (300, 200), (150, 150, 150), 1)           # стол — не окно
    cv2.line(img, (200, 206), (300, 206), (150, 150, 150), 1)
    # участок сплошной стены над проёмом: светлая кайма сверху и притолока снизу — не окно
    cv2.rectangle(img, (420, 30), (470, 38), (0, 0, 0), -1)
    cv2.rectangle(img, (400, 28), (419, 40), (0, 0, 0), -1)
    cv2.rectangle(img, (471, 28), (490, 40), (0, 0, 0), -1)
    cv2.line(img, (420, 29), (470, 29), (180, 180, 180), 1)
    cv2.line(img, (420, 41), (470, 41), (180, 180, 180), 1)
    # лестничный марш на серой заливке между стенками: ступени через 3 px — не окна
    cv2.rectangle(img, (425, 160), (469, 260), (205, 205, 205), -1)
    cv2.rectangle(img, (420, 160), (424, 260), (0, 0, 0), -1)
    cv2.rectangle(img, (470, 160), (474, 260), (0, 0, 0), -1)
    for y in range(165, 256, 3):
        cv2.line(img, (425, y), (469, y), (120, 120, 120), 1)
    return img


def test_glazing_windows_found_between_piers_not_furniture():
    import cv2
    from fp.windows_cv import find_glazing
    bands = find_glazing(cv2.cvtColor(_window_sheet(), cv2.COLOR_BGR2GRAY), 50.0)
    assert len(bands) == 3 and all(abs(b["length_m"] - 1.0) <= 0.15 for b in bands)
    assert all(b["bbox"][1] < 120 for b in bands)


def test_glazing_windows_join_model_windows():
    from fp.infer import add_glazing_windows
    from fp.query import count
    from fp.scene import Scene
    sc = Scene.build("x", 600, 300, [("window", [(100, 98), (150, 98), (150, 113), (100, 113)], 0.5)],
                     meta={"px_per_m": 50.0})
    info = add_glazing_windows(_window_sheet(), sc)
    assert info["added"] == 2 and info["confirmed"] == ["win1"]
    a = count(sc, "window")
    assert a.answer == 3 and a.confidence == "medium" and "остеклен" in a.note


def test_rectify_room_snaps_ragged_model_contour_to_walls():
    import cv2
    import numpy as np
    from fp.rooms_cv import rectify_room
    img = np.full((300, 400), 255, np.uint8)
    cv2.rectangle(img, (50, 50), (350, 250), 0, 6)                     # стены комнаты 300×200
    # контур модели: заходит за правую стену ступенькой и не доходит до верхней
    ragged = [[56, 60], [330, 60], [362, 120], [362, 160], [344, 165], [344, 244], [56, 244]]
    r = rectify_room(img, ragged, 50.0)
    assert r is not None
    xs, ys = [p[0] for p in r], [p[1] for p in r]
    assert abs(min(xs) - 54) <= 2 and abs(max(xs) - 346) <= 2 and abs(min(ys) - 54) <= 2 and abs(max(ys) - 246) <= 2
    l_shape = [[56, 56], [344, 56], [344, 120], [150, 120], [150, 244], [56, 244]]   # Г-образный — не трогаем
    assert rectify_room(img, l_shape, 50.0) is None


def test_count_falls_back_to_inventory_when_model_finds_none(monkeypatch):
    from fp import query
    from fp.scene import Scene
    sc = Scene.build("x", 500, 500, [])
    sc.meta["inventory"] = {"windows": 96, "doors": 1}
    monkeypatch.setattr(query, "ROUTE", {"window": "yolo", "door": "yolo"})
    a = query.count(sc, "window")
    assert a.answer == 96 and a.confidence == "low" and "описи" in a.note
    assert query.count(sc, "door").answer == 0          # опись видит одну — это не повод отвечать по ней


def _pipe_sheet():
    """Лист 800×400, 50 px/м: труба 8" (двойная линия, 11 px) длиной 8 м с разрывом на клапан, стена из двух линий
    по границе помещения, подписи читает «ИИ» (подставлены)."""
    import cv2
    import numpy as np
    img = np.full((400, 800, 3), 255, np.uint8)
    for y in (150, 161):                                    # труба: две линии через 11 px = 0.22 м ≈ 8"
        cv2.line(img, (100, y), (380, y), (0, 0, 0), 1)
        cv2.line(img, (400, y), (500, y), (0, 0, 0), 1)     # разрыв 20 px — клапан
    for y in (40, 50):                                      # стена из двух линий по верху помещения
        cv2.line(img, (20, y), (780, y), (0, 0, 0), 1)
    cv2.line(img, (600, 300), (750, 300), (0, 0, 0), 1)     # малая труба 2" одной линией, 3 м
    return img


def test_piping_lengths_by_label_and_fittings_by_tag():
    from fp.piping import build
    from fp.query import fittings, pipes
    from fp.scene import Scene
    sc = Scene.build("x", 800, 400, [("room_other", [(20, 45), (780, 45), (780, 390), (20, 390)], 0.6)],
                     meta={"px_per_m": 50.0})
    items = [{"text": '8"', "kind": "pipe_size", "size_in": 8, "x": 200, "y": 120, "target_x": 200, "target_y": 156},
             {"text": '2"', "kind": "pipe_size", "size_in": 2, "x": 650, "y": 280, "target_x": 650, "target_y": 300},
             {"text": "CV-1", "kind": "valve", "size_in": 0, "x": 390, "y": 130, "target_x": 390, "target_y": 156},
             {"text": "CV-2", "kind": "valve", "size_in": 0, "x": 450, "y": 130, "target_x": 450, "target_y": 156},
             {"text": "PG", "kind": "instrument", "size_in": 0, "x": 300, "y": 200, "target_x": 300, "target_y": 200},
             {"text": "PS-1", "kind": "support", "size_in": 0, "x": 250, "y": 190, "target_x": 250, "target_y": 161},
             {"text": "PS-1", "kind": "support", "size_in": 0, "x": 450, "y": 190, "target_x": 450, "target_y": 161}]
    sc.meta["piping"] = build(_pipe_sheet(), sc, items)
    by = sc.meta["piping"]["by_size_m"]
    assert abs(by["8"] - 7.6) <= 0.3          # 280 + 100 px = 7.6 м, разрыв на клапан — та же труба
    assert abs(by["2"] - 3.0) <= 0.2
    assert sum(sc.meta["piping"]["unlabeled_by_width_m"].values()) < 1.0   # стена из двух линий — не труба (15 м)
    assert pipes(sc, 8).answer["total_m"] == by["8"]
    f = fittings(sc).answer
    assert f["CV"]["count"] == 2 and f["PG"]["count"] == 1 and f["PS"]["count"] == 2
    assert fittings(sc, "valve").answer.keys() == {"CV"}


def test_partition_from_model_rooms_is_not_a_pipe():
    """M-101A: ИИ дал одну рамку на всё здание, перегородки туалета известны только по контурам модели
    (meta.yolo_room_polygons) — их двойные линии не трубы, даже если выноска подписи ближе к ним."""
    import cv2
    import numpy as np
    from fp.piping import build
    from fp.scene import Scene
    img = np.full((400, 800, 3), 255, np.uint8)
    for x in (300, 311):                                  # перегородка: две линии через 11 px
        cv2.line(img, (x, 60), (x, 380), (0, 0, 0), 1)
    for x in (500, 511):                                  # труба 8" в зале
        cv2.line(img, (x, 60), (x, 380), (0, 0, 0), 1)
    big = [(20, 45), (780, 45), (780, 390), (20, 390)]
    items = [{"text": '8"', "kind": "pipe_size", "size_in": 8, "x": 400, "y": 200, "target_x": 306, "target_y": 200},
             {"text": '8"', "kind": "pipe_size", "size_in": 8, "x": 600, "y": 200, "target_x": 505, "target_y": 200}]
    sc = Scene.build("x", 800, 400, [("room_other", big, 0.6)], meta={"px_per_m": 50.0})
    assert build(img, sc, items)["by_size_m"]["8"] > 12          # без контуров модели перегородка — «труба»
    sc.meta["yolo_room_polygons"] = [[[20, 45], [300, 45], [300, 390], [20, 390]],
                                     [[311, 45], [780, 45], [780, 390], [311, 390]]]
    p = build(img, sc, items)
    assert abs(p["by_size_m"]["8"] - 6.4) <= 0.3                 # только труба: 320 px / 50
    assert all(s["bbox"][0] > 400 for s in p["segments"])


def test_exterior_wall_is_not_a_pipe():
    """M-101A: выноска «2" FILTER SEPARATOR…» легла на наружную стену — стена (улица с одной стороны) не труба;
    труба снаружи здания (улица с обеих сторон) и внутри (ни с одной) остаются."""
    import cv2
    import numpy as np
    from fp.piping import build
    from fp.scene import Scene
    img = np.full((500, 900, 3), 255, np.uint8)
    for d in (0, 4):                                        # наружная стена из двух линий через 4 px
        cv2.rectangle(img, (100 - d, 100 - d), (700 + d, 400 + d), (0, 0, 0), 1)
    for y in (200, 210):                                    # труба внутри здания
        cv2.line(img, (200, y), (600, y), (0, 0, 0), 1)
    for x in (800, 810):                                    # труба снаружи, вдоль здания
        cv2.line(img, (x, 120), (x, 380), (0, 0, 0), 1)
    items = [{"text": '2"', "kind": "pipe_size", "size_in": 2, "x": 400, "y": 60, "target_x": 400, "target_y": 98},
             {"text": '8"', "kind": "pipe_size", "size_in": 8, "x": 400, "y": 250, "target_x": 400, "target_y": 205},
             {"text": '8"', "kind": "pipe_size", "size_in": 8, "x": 850, "y": 250, "target_x": 805, "target_y": 250}]
    sc = Scene.build("x", 900, 500, [], meta={"px_per_m": 50.0, "sheet": {"drawing_bbox": [20, 20, 880, 480]}})
    p = build(img, sc, items)
    assert "2" not in p["by_size_m"] or p["by_size_m"]["2"] == 0     # стена не стала трубой 2"
    assert abs(p["by_size_m"]["8"] - 13.2) <= 0.4                    # 400 + 260 px = 13.2 м: обе трубы на месте


def test_no_piping_explains_why():
    """Старая сцена листа трубопроводов (без meta.piping) — не «это не лист трубопроводов», а «разберите заново»."""
    from fp.query import fittings, pipes
    from fp.scene import Scene
    piping_sheet = {"title_block": {"drawing_title": "PIPING PLAN", "sheet_number": "M-101A"}}
    old = Scene.build("x", 10, 10, [], meta={"sheet": piping_sheet})
    assert "заново" in pipes(old).note and "не лист" not in pipes(old).note
    assert "заново" in fittings(old, "CV").note
    from fp.chat import chat
    reply = chat(old, "покажи трубы", use_llm=False)["reply"]
    assert reply.startswith("Лист M-101A") and "null" not in reply
    small = Scene.build("x", 10, 10, [], meta={"sheet": piping_sheet, "piping": {"error": "масштаб 9.9 px/м мельче 33"}})
    assert "масштаб" in pipes(small).note
    plan = Scene.build("x", 10, 10, [], meta={"sheet": {"title_block": {"drawing_title": "FLOOR PLAN",
                                                                         "sheet_number": "A-2"}}})
    assert "не лист трубопроводов" in pipes(plan).note


def test_scene_save_handles_numpy_values(tmp_path):
    import numpy as np
    from fp.scene import Scene
    sc = Scene.build("x", 10, 10, [])
    sc.meta["piping"] = {"a0": np.int32(5), "w": np.float64(1.5), "arr": np.array([1, 2])}
    sc.save(tmp_path)
    assert Scene.load(tmp_path).meta["piping"] == {"a0": 5, "w": 1.5, "arr": [1, 2]}


def test_room_name_from_schedule_is_shown_by_rules():
    from fp.chat import chat
    from fp.scene import Scene
    sq = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    sc = Scene.build("x", 500, 500, [("toilet", sq(10, 10, 20), 0.97)])
    mk = lambda i, n, a: {"id": f"sp{i}", "name": n, "ok": True, "area_m2": a, "area_source": "контур модели",
                          "baseboard_m": 10.0, "baseboard_note": None, "confidence": "medium"}
    sc.meta["spaces"] = [mk(1, "Multi-Purpose Hall", 237.8), mk(2, "Group Room", 20.0), mk(3, "Staff Room", 15.0),
                         mk(4, "Toilets", 12.0)]
    r = chat(sc, "Multi-Purpose Hall", use_llm=False)
    assert r["parsed_by"] == "cache" and r["calls"][0]["tool"] == "rooms" and "237.8" in r["reply"]
    assert r["action"] == {"type": "highlight", "ids": ["sp1"]}
    g = chat(sc, "покажи Group Room", use_llm=False)
    assert g["action"]["ids"] == ["sp2"]                     # не все «…Room»
    t = chat(sc, "how many toilets", use_llm=False)
    assert t["calls"][0]["tool"] == "count"                  # подсчёт унитазов остаётся подсчётом


def test_drawing_sheet_number_long_form():
    """Школа C01: ИИ иногда читает вместо «C01» длинный номер SEN-DFE-XX-00-D-X-0201 — это тоже чертёжный лист;
    квартиры (номера нет или «3») — нет."""
    from fp.infer import is_drawing_sheet
    for s in ("M-101A", "A-2", "C01", "S-100F", "SEN-DFE-XX-00-D-X-0201"):
        assert is_drawing_sheet(s), s
    for s in ("", None, "3", "12", "Plan 2", "KITCHEN"):
        assert not is_drawing_sheet(s), s


def test_glossary_rooms_in_many_languages():
    """Словарь обозначений (fp/glossary.json): сокращения финских квартир, полные названия на 15 языках,
    длинное совпадение важнее короткого, составные подписи — оба вида по порядку."""
    from fp.glossary import lookup, lookup_all, room_class
    cases = {"MH2 12.5 m²": "bedroom", "K": "kitchen", "KH": "bathroom", "PH": "bathroom", "S": "sauna", "VH": "dressing",
             "PARVEKE": "balcony", "KHH": "laundry", "SOVRUM": "bedroom", "Elternschlafzimmer": "bedroom", "SDB": "bathroom",
             "Cocina": "kitchen", "Łazienka": "bathroom", "Rõdu": "balcony", "Санузел": "wc", "Лоджия": "balcony",
             "Multi-Purpose Hall": "assembly_hall", "Entrance Hall": "entry", "Group Room": "classroom", "Stair B": "stair",
             "Accessible Toilet": "wc", "Walk-in closet": "dressing"}
    for label, kind in cases.items():
        assert (lookup(label) or {}).get("kind") == kind, label
    assert [m["kind"] for m in lookup_all("K/OH")] == ["kitchen", "living"]
    assert room_class("Secondary Classroom") == "room_other" and room_class("VH") == "room_storage"
    assert lookup("zzz qqq") is None and room_class(None) == "room_other"


def test_glossary_user_term(tmp_path, monkeypatch):
    """Дополнение пользователя — отдельный файл, сразу участвует в поиске; неизвестный вид — ошибка."""
    import pytest
    from fp import glossary
    monkeypatch.setattr(glossary, "USER", tmp_path / "glossary_user.json")
    glossary.reload()
    assert glossary.lookup("XYZ") is None
    assert glossary.add_term("XYZ", "dressing")["kind"] == "dressing"
    assert glossary.lookup("xyz 3")["kind"] == "dressing"
    with pytest.raises(ValueError):
        glossary.add_term("Q1", "no-such-kind")
    monkeypatch.undo()
    glossary.reload()


def test_cache_recreates_tables_after_file_removed(tmp_path):
    """Сброс кэша при работающем сервере (файл перенесли) — запрос не падает «no such table: answers»."""
    from fp import cache
    p = tmp_path / "c.sqlite"
    cache.configure(p)
    p.unlink()
    assert cache.get_answer("s", "q") is None
    cache.configure(None)


def test_user_corrections_change_answers_and_undo(tmp_path, monkeypatch):
    """Исправления из чата (fp/corrections.py): тип помещения, лишний объект, число со слов пользователя,
    обозначение в словарь; всё в журнале, отмена возвращает как было."""
    from fp import glossary
    from fp.query import call
    from fp.scene import Scene
    monkeypatch.setattr(glossary, "USER", tmp_path / "glossary_user.json")
    glossary.reload()
    sq = lambda x0, y0, s=10: [[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]]
    sc = Scene.build("x", 400, 200, [("room_bedroom", sq(0, 0, 100), 0.6), ("door", sq(50, 95), 0.7),
                                      ("door", sq(150, 95), 0.7), ("window", sq(10, 0), 0.8)])
    bed = next(r.id for r in sc.rooms)
    doors = [o.id for o in sc.objects if o.cls == "door"]
    assert call(sc, "count", {"cls": "door"}).answer == 2

    a = call(sc, "set_room_type", {"room": bed, "type": "other"})
    assert a.answer["now"] == "other" and sc.rooms[0].cls == "room_other" and sc.meta.pop("_dirty")
    call(sc, "remove_objects", {"ids": [doors[1]]})
    assert call(sc, "count", {"cls": "door"}).answer == 1
    a = call(sc, "set_count", {"cls": "window", "value": 5})
    assert call(sc, "count", {"cls": "window"}).answer == 5 and call(sc, "count", {"cls": "window"}).confidence == "high"
    call(sc, "define_term", {"term": "QX", "kind": "dressing"})
    assert glossary.lookup("QX 2")["kind"] == "dressing"
    assert len(call(sc, "list_corrections", {}).answer) == 4

    call(sc, "undo_correction", {})          # define_term: словарь остаётся, журнал короче
    call(sc, "undo_correction", {})          # set_count
    assert call(sc, "count", {"cls": "window"}).answer == 1
    call(sc, "undo_correction", {})          # remove_objects — дверь вернулась
    assert call(sc, "count", {"cls": "door"}).answer == 2
    call(sc, "undo_correction", {})          # set_room_type
    assert sc.rooms[0].cls == "room_bedroom"
    assert call(sc, "undo_correction", {}).note == "на этом плане исправлений нет"
    monkeypatch.undo()
    glossary.reload()


def test_correction_tools_are_offered_to_llm_and_not_cached():
    from fp.query import CORRECTION_TOOLS, TOOLS
    names = {t["name"] for t in TOOLS}
    assert {"define_term", "set_room_type", "remove_objects", "set_count", "undo_correction"} <= names
    assert CORRECTION_TOOLS <= names
    kinds = next(t for t in TOOLS if t["name"] == "define_term")["input_schema"]["properties"]["kind"]["enum"]
    assert "dressing" in kinds and "bedroom" in kinds


def test_tiled_inventory_confidence_on_big_sheet():
    """Большой лист с описью по фрагментам: отвечают детекции, согласие — с допуском; квартиры — точное совпадение."""
    from fp.query import _agree
    assert _agree(131, 103, True) and _agree(131, 75, True) is False and _agree(83, 65, True) and _agree(3, 4, True)
    assert _agree(3, 4, False) is False and _agree(5, 5, False)


def test_tiled_inventory_keeps_points_in_tile_core(monkeypatch):
    """Объект на перекрытии двух фрагментов засчитывается один раз — тому, в чьём ядре его точка."""
    import types
    import numpy as np
    from fp import inventory_tiles as IT
    img = np.full((2000, 3000), 255, np.uint8)
    img[::50, :] = 0                                   # не пустая бумага
    calls = []

    class Msgs:
        def create(self, **kw):
            calls.append(kw)
            # каждый фрагмент «видит» дверь в одной и той же точке листа: x=1000/s, y=200/s в px full
            return types.SimpleNamespace(stop_reason="end_turn", usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
                                         content=[types.SimpleNamespace(type="text", text='{"items": [], "room_labels": []}')])
    monkeypatch.setattr("anthropic.Anthropic", lambda: types.SimpleNamespace(messages=Msgs()))
    inv, _ = IT.read_inventory_tiled(img, IT.WORK_PPM, None)     # масштаб 1:1
    assert inv["method"] == "tiles" and inv["tiles"] == len(calls) and inv["doors"] == 0 and inv["floors"] == 1


def test_clear_cache_removes_scenes(tmp_path, monkeypatch):
    """Кнопка «Очистить кэш»: планы и кэш ответов удаляются, резервных копий не остаётся."""
    import json
    import server.app as A
    scenes = tmp_path / "scenes"
    (scenes / "plan-1").mkdir(parents=True)
    (scenes / "plan-1" / "scene.json").write_text(json.dumps({}))
    (scenes / "cache.sqlite").write_text("")
    monkeypatch.setattr(A, "SCENES", scenes)
    assert A.clear_cache() == {"removed": 2, "plans": 1}
    assert not any(scenes.iterdir()) and [p.name for p in tmp_path.iterdir()] == ["scenes"]
    assert A.clear_cache() == {"removed": 0, "plans": 0}


def test_rooms_pass_uses_room_model_and_coarse_tiles(monkeypatch):
    """Второй проход по комнатам: своя модель (Opus через beta), крупнее и меньше фрагментов, считает только комнаты."""
    import types
    import numpy as np
    from fp import inventory_tiles as IT
    img = np.full((4000, 6000), 255, np.uint8)
    img[::50, :] = 0
    calls = []
    ans = '{"items": [{"type": "kitchen", "x": 10, "y": 10}], "room_labels": []}'

    class Msgs:
        def create(self, **kw):
            calls.append(kw)
            return types.SimpleNamespace(stop_reason="end_turn", usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
                                         content=[types.SimpleNamespace(type="text", text=ans)])
    client = types.SimpleNamespace(messages=Msgs(), beta=types.SimpleNamespace(messages=Msgs()))
    monkeypatch.setattr("anthropic.Anthropic", lambda: client)
    rr, _ = IT.read_rooms_tiled(img, 80.0, None)       # 80 px/м -> 40 px/м: 3000x2000, ядро 1200 -> 3x2 фрагмента;
    # точка (10, 10) в ядре только у первого фрагмента — у остальных она в запасе: кухня одна
    assert rr["tiles"] == 6 == len(calls) and all(c["model"] == IT.ROOM_MODEL for c in calls)
    assert all(c.get("betas") for c in calls) == IT.ROOM_MODEL.startswith("claude-opus")
    assert rr["counts"]["kitchens"] == 1 and set(rr["counts"]) == set(IT.ROOM_TYPES.values())
    n_small, _ = len(calls), IT.read_inventory_tiled(img, 80.0, None)
    assert len(calls) - n_small > rr["tiles"]          # Haiku-проход мелче: фрагментов больше


def test_upload_cost_breakdown():
    """Цена первичного разбора для интерфейса: итог, шаги с ненулевой ценой, время."""
    import server.app as A
    meta = {"sheet_cost_usd": 0.028, "analysis_s": 41.3, "scale": {"vlm_cost_usd": 0.0021},
            "door_verify": {"cost_usd": 0.0075}, "inventory": {"cost_usd": 0.0179}, "fixture_check": {"cost_usd": 0.0}}
    r = A._cost(meta)
    assert r["cost_usd"] == 0.028 and r["analysis_s"] == 41.3
    assert r["cost_parts"] == {"чтение листа": 0.0021, "проверка дверей": 0.0075, "опись": 0.0179}
    assert A._cost({})["cost_usd"] == 0.0
