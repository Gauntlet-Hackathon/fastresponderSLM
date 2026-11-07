from rescue_tagger.safety import cue_kinds, review_flags


def _pay(req=True, conf=0.95, hazards=(), vulnerable=()):
    return {"is_request": {"value": req, "confidence": conf},
            "hazards": [{"kind": k, "confidence": c} for k, c in hazards],
            "vulnerable": [{"kind": k, "confidence": c} for k, c in vulnerable]}


def test_cues_cover_three_languages_and_misspellings():
    assert {"trapped", "rising_water"} <= cue_kinds("trapped in attic, 2 adults")
    assert "trapped" in cue_kinds("estamos atrapados en la casa")
    assert "trapped" in cue_kinds("nhà tôi bị mắc kẹt")
    assert "trapped" in cue_kinds("nha toi bi mac ket")           # typed without diacritics
    assert "elderly" in cue_kinds("watr to the porch 2 old peopel")
    assert "child" in cue_kinds("my son's ventilator is dying")
    assert "fire" in cue_kinds("huele a humo")
    assert "on_roof" in cue_kinds("đang ở trên mái nhà")
    assert "child" not in cue_kinds("Sonia's song")                # whole-word cues stay whole


def test_flags_a_cue_the_model_missed_but_not_one_it_reported():
    text = "trapped in attic 880 Reed Rd, 2 adults 1 child"
    flags = review_flags(text, _pay(hazards=[("trapped", 0.9)], vulnerable=[("child", 0.8)]))
    assert flags == ["cue:rising_water"]
    assert review_flags(text, _pay(hazards=[("trapped", 0.9), ("rising_water", 0.7)],
                                   vulnerable=[("child", 0.8)])) == []


def test_near_miss_and_unsure_request_are_flagged():
    assert "near:fire" in review_flags("maybe smoke??", _pay(hazards=[("fire", 0.41)]))
    assert review_flags("hello?", _pay(conf=0.5)) == ["unsure_request"]


def test_confident_non_request_is_not_flagged_by_cue_words():
    assert review_flags("Shelter open for families with kids, water rising on I-10",
                        _pay(req=False, conf=0.95)) == []


def test_short_vietnamese_cues_need_their_accents():
    for text in ("no tenemos luz", "vivo con mi esposo", "better than nothing", "Tim is here",
                 "my 5 year old daughter", "this fall", "he got fired"):
        kinds = cue_kinds(text)
        assert not kinds & {"fire", "medical", "injury", "disabled", "elderly"}, (text, kinds)
    assert cue_kinds("vivo con mi esposo") == set()
    assert "child" in cue_kinds("my 5 year old daughter")
    assert "fire" in cue_kinds("nghe tiếng nổ")


def test_decomposed_accents_are_normalised():
    import unicodedata
    assert "trapped" in cue_kinds(unicodedata.normalize("NFD", "chúng tôi bị mắc kẹt"))
